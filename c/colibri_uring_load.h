/* io_uring expert-load batches (Linux, URING=1): one owner prepares all reads of a block, submits
 * them in one syscall and reaps completions on demand. Wraps the raw ring in uring.h.
 *
 * Not a standalone header: colibri.c includes it in place, and the code below uses that
 * file's types and statics (Model, ESlot, QT, now_s, ...). Moving it out changes no
 * behaviour; the translation unit is byte-for-byte the same after preprocessing. */
#ifndef COLIBRI_URING_LOAD_H
#define COLIBRI_URING_LOAD_H

#ifdef __linux__
/* io_uring expert batches.  One owner prepares all reads for a block, submits
 * them in one syscall, and reaps CQEs on demand.  The kernel, rather than a set
 * of blocking pthreads, owns the I/O concurrency. */
#define URING_LOAD_MAX 64
#define URING_REQ_MAX  512
typedef struct {
    int load, expect;
    int fd, primary_fd, rep;
    void *buf;
    int64_t off;
} UringRead;
typedef struct {
    Model *m; ESlot *s; int layer,eid,fatal;
    st_tensor *tw[3],*tq[3]; int64_t pos[3];
    int pending,done,finalized,error;
} UringLoad;
typedef struct {
    ColiUring ring;
    UringLoad load[URING_LOAD_MAX];
    UringRead req[URING_REQ_MAX];
    int nload,nreq,started;
} UringBatch;
static UringBatch g_ub_pipe, g_ub_pilot;

static int uring_batch_init(UringBatch *b){
    if(b->started) return 0;
    if(coli_uring_init(&b->ring,URING_REQ_MAX)) return -1;
    b->started=1; return 0;
}
static void uring_batch_reset(UringBatch *b){
    b->nload=0; b->nreq=0;
}
static int uring_load_error(UringLoad *l,int err,const char *what){
    l->error=err?err:EIO; l->done=1;
    if(l->fatal){ errno=l->error; perror(what); exit(1); }
    return -1;
}
static int uring_add_read(UringBatch *b,int li,int fd,int primary_fd,int rep,
                          void *buf,size_t len,int64_t off,size_t expect){
    if(b->nreq>=URING_REQ_MAX || expect>INT_MAX){ errno=E2BIG; return -1; }
    if(rep<0 || rep>=MIR_REPS){ errno=EINVAL; return -1; }
    int ri=b->nreq++;
    b->req[ri]=(UringRead){li,(int)expect,fd,primary_fd,rep,buf,off};
    if(coli_uring_prep_read(&b->ring,fd,buf,len,off,(uint64_t)ri+1)) return -1;
    b->load[li].pending++;
    return 0;
}
/* Buffered replica read with per-shard fallback.  Keep the actual source in
 * UringRead: completion accounting, DROP and runtime error fallback must all
 * describe the fd that was really submitted, not recompute a route later. */
static int uring_add_rep_read(UringBatch *b,int li,shards *S,int primary_fd,
                              int rep,void *buf,size_t len,int64_t off,
                              size_t expect){
    int fd=st_fd_rep(S,primary_fd,rep);
    int used=(rep && fd>=0)?rep:0;
    if(fd<0) fd=primary_fd;
    return uring_add_read(b,li,fd,primary_fd,used,buf,len,off,expect);
}
/* Returns the load index. URING is intentionally a quantized streaming path;
 * unsupported layouts fail instead of silently dropping back to pread. */
static int uring_load_add(UringBatch *b,Model *m,int layer,int eid,ESlot *s,int fatal){
    if(b->nload>=URING_LOAD_MAX){ errno=E2BIG; return -1; }
    int li=b->nload++;
    UringLoad *l=&b->load[li]; memset(l,0,sizeof(*l));
    l->m=m; l->s=s; l->layer=layer; l->eid=eid; l->fatal=fatal;
    char nm[3][288],qn[320]; const char suf[3][16]={"gate_proj","up_proj","down_proj"};  /* bounded suf: see #484 */
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(g_mmap || !st_has(&m->S,qn))
        return uring_load_error(l,ENOTSUP,"URING requires quantized expert tensors"),li;
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    for(int k=0;k<3;k++){
        l->tw[k]=st_find(&m->S,nm[k]);
        size_t n=strnlen(nm[k],sizeof(nm[k]));
        if(n+3>=sizeof(qn)) return uring_load_error(l,ENAMETOOLONG,"io_uring expert metadata"),li;
        memcpy(qn,nm[k],n); memcpy(qn+n,".qs",4); l->tq[k]=st_find(&m->S,qn);
        if(!l->tw[k]||!l->tq[k]) return uring_load_error(l,ENOENT,"io_uring expert metadata"),li;
    }
    int64_t wtot=l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes;
    int64_t ftot=(l->tq[0]->nbytes+l->tq[1]->nbytes+l->tq[2]->nbytes)/4;
    if(wtot<=0 || ftot<=0) return uring_load_error(l,EINVAL,"io_uring expert size"),li;
    /* Same shrink rule as the pread path (#856): a slot that once held a wider
     * expert must not carry that width into a narrower row, or the per-row cap
     * cap_for_ram() computes stops being true. Same hysteresis, same arena guard. */
    int64_t want=wtot+8192, hyst=want/4; if(hyst<(1<<16)) hyst=1<<16;
    int64_t fhyst=ftot/4; if(fhyst<(1<<14)) fhyst=1<<14;   /* floats, so 64 KB again */
    if(!s->slab || want>s->slab_cap || (g_slab_shrink && !s->aslab && s->slab_cap > want+hyst)){
#ifdef COLI_METAL
        if(s->slab&&g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=need; if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,(size_t)wtot+8192)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=wtot+8192;
#endif
    }
    if(!s->fslab || ftot>s->fslab_cap || (g_slab_shrink && !s->afslab && s->fslab_cap > ftot+fhyst)){
#ifdef COLI_METAL
        if(s->fslab&&g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab); size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->fslab,16384,fb)){
            s->fslab=NULL; s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot; if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab); s->fslab=malloc((size_t)ftot*sizeof(float));
        if(!s->fslab){ s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot;
#endif
    }
    /* DUAL-SSD (#1165): pick this expert's replica exactly as the blocking
     * path does. The uring path used the primary fd unconditionally, so
     * URING=1 read every expert from drive 0 and COLI_MODEL_MIRROR bought
     * nothing -- visible as an idle mirror in iostat while URING=0 lit it up.
     * Same deterministic hash of (layer,eid), so an expert always comes from
     * the same drive and PILOT's readahead lands where the demand read will.
     * Partial mirrors fall back to the primary per shard, as at the blocking
     * site. */
    int rep=expert_route(layer,eid);
    if(rep && st_fd_rep(&m->S,l->tw[0]->fd,rep)<0) rep=0;
    int ord[3]={0,1,2};
    for(int a=0;a<3;a++) for(int z=a+1;z<3;z++) if(l->tw[ord[z]]->off<l->tw[ord[a]]->off){int t=ord[a];ord[a]=ord[z];ord[z]=t;}
    int contig=l->tw[ord[0]]->fd==l->tw[ord[1]]->fd && l->tw[ord[1]]->fd==l->tw[ord[2]]->fd
        && l->tw[ord[0]]->off+l->tw[ord[0]]->nbytes==l->tw[ord[1]]->off
        && l->tw[ord[1]]->off+l->tw[ord[1]]->nbytes==l->tw[ord[2]]->off;
    if(contig){
        int64_t off0=l->tw[ord[0]]->off;
        int dfd=g_direct?st_direct_fd_rep(&m->S,l->tw[ord[0]]->fd,rep):-1;
        if(dfd>=0){
            int64_t base=off0&~4095LL,need=(off0-base)+wtot,len=(need+4095)&~4095LL;
            l->pos[ord[0]]=off0-base; l->pos[ord[1]]=l->pos[ord[0]]+l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_read(b,li,dfd,l->tw[ord[0]]->fd,rep,
                              s->slab,(size_t)len,base,(size_t)need))
                return uring_load_error(l,errno,"io_uring direct expert read"),li;
        }else{
            l->pos[ord[0]]=0; l->pos[ord[1]]=l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_rep_read(b,li,&m->S,l->tw[ord[0]]->fd,rep,
                                  s->slab,(size_t)wtot,off0,(size_t)wtot))
                return uring_load_error(l,errno,"io_uring expert read"),li;
        }
    }else{
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a]; l->pos[k]=o;
            if(uring_add_rep_read(b,li,&m->S,l->tw[k]->fd,rep,
                                  s->slab+o,(size_t)l->tw[k]->nbytes,
                                  l->tw[k]->off,(size_t)l->tw[k]->nbytes))
                return uring_load_error(l,errno,"io_uring expert read"),li;
            o+=l->tw[k]->nbytes;
        }
    }
    int64_t fo=0;
    for(int k=0;k<3;k++){
        if(uring_add_rep_read(b,li,&m->S,l->tq[k]->fd,rep,
                              s->fslab+fo,(size_t)l->tq[k]->nbytes,
                              l->tq[k]->off,(size_t)l->tq[k]->nbytes))
            return uring_load_error(l,errno,"io_uring expert scale read"),li;
        fo+=l->tq[k]->nbytes/4;
    }
    return li;
}
static void uring_reap(UringBatch *b){
    struct io_uring_cqe cqe;
    while(coli_uring_peek(&b->ring,&cqe)){
        if(!cqe.user_data || cqe.user_data>(uint64_t)b->nreq) continue;
        UringRead *r=&b->req[cqe.user_data-1]; UringLoad *l=&b->load[r->load];
        int served=cqe.res>=r->expect;
        if(!served && r->rep){
            /* Match mir_pread's availability contract. A missing replica file
             * is handled before submission; an I/O error or short completion
             * discovered here gets one synchronous retry on the primary. This
             * path is exceptional, so preserving inference is more important
             * than retaining queue depth while a mirror is degraded. */
            static _Atomic int warned;
            if(!atomic_exchange(&warned,1))
                fprintf(stderr,"[MIRROR] io_uring read error on the mirror copy — falling back to the primary drive\n");
            if(!pread_full(r->primary_fd,r->buf,r->expect,r->off,
                           "io_uring mirror fallback")){
                r->fd=r->primary_fd; r->rep=0; cqe.res=r->expect; served=1;
            }
        }
        if(served){
            atomic_fetch_add_explicit(&g_mir_bytes[r->rep],cqe.res,memory_order_relaxed);
            atomic_fetch_add_explicit(&g_mir_nread[r->rep],1,memory_order_relaxed);
        }else if(!l->error) l->error=cqe.res<0?-cqe.res:EIO;
        if(l->pending>0) l->pending--;
        if(l->pending==0) l->done=1;
    }
}
static int uring_submit_batch(UringBatch *b){
    if(coli_uring_enter(&b->ring,0)<0) return -1;
    uring_reap(b); return 0;
}
static int uring_wait_load(UringBatch *b,int li){
    UringLoad *l=&b->load[li];
    while(!l->done){
        uring_reap(b); if(l->done) break;
        if(coli_uring_enter(&b->ring,1)<0) return uring_load_error(l,errno,"io_uring wait");
    }
    return l->error?-1:0;
}
static int uring_finalize_load(UringBatch *b,int li,int publish_eid){
    UringLoad *l=&b->load[li]; ESlot *s=l->s;
    if(l->finalized) return 0;
    if(uring_wait_load(b,li)<0){ errno=l->error; if(l->fatal){perror("io_uring expert completion");exit(1);} return -1; }
    if(g_drop){
        /* Each request remembers its final source (mirror, per-shard primary,
         * or primary after an error retry). Dropping tensor metadata's primary
         * fd here left the pages actually read from a mirror resident forever. */
        for(int ri=0;ri<b->nreq;ri++) if(b->req[ri].load==li)
            posix_fadvise(b->req[ri].fd,b->req[ri].off,
                          (off_t)b->req[ri].expect,POSIX_FADV_DONTNEED);
    }
    Cfg *c=&l->m->c; int I=c->moe_inter,D=c->hidden; float *fp[3]; int64_t fo=0;
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D},II[3]={D,D,I};
    for(int k=0;k<3;k++){
        fp[k]=s->fslab+fo; fo+=l->tq[k]->nbytes/4;
        int64_t nb=l->tw[k]->nbytes;
        /* qt_resolve_fmt like the other two expert paths: the raw ?1:?2:3 inference here
         * missed grouped int4 (fmt=4, gs never set) and would mis-tag int3-g64 as int2. */
        int gs=0;
        int fmt=qt_resolve_fmt(l->tw[k]->name,OO[k],II[k],nb,l->tq[k]->nbytes,&gs,NULL);   /* routed expert: never stamped */
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+l->pos[k]); qt[k]->q4=s->slab+l->pos[k]; qt[k]->s=fp[k];
    }
    atomic_fetch_add_explicit(&g_prof_io,
        l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes+fo*4,
        memory_order_relaxed);
    if(publish_eid) s->eid=l->eid;
    l->finalized=1; return 0;
}
static int uring_wait_all(UringBatch *b){
    for(int i=0;i<b->nload;i++) if(uring_wait_load(b,i)<0) return -1;
    return 0;
}
#endif

#endif /* COLIBRI_URING_LOAD_H */
