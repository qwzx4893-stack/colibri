/* Local expert-worker cluster: the wire protocol, the coordinator-side client (cluster_moe_batch)
 * and the worker loop (cluster_worker_run). Off unless workers are configured; POSIX only.
 *
 * Not a standalone header: glm52.c includes it in place, and the code below uses that
 * file's types and statics (Model, ESlot, QT, now_s, ...). Moving it out changes no
 * behaviour; the translation unit is byte-for-byte the same after preprocessing. */
#ifndef GLM52_CLUSTER_H
#define GLM52_CLUSTER_H

#if !defined(_WIN32)
/* Expert-worker protocol. Headers use network-order u32 values; activation
 * bytes remain raw little-endian f32, matching the native engine ABI. One
 * request contains the routed batch-union for a layer. */
#define LMA_CLUSTER_MAGIC "COLIEX01"
#define LMA_CLUSTER_VERSION 1u
static int cluster_io(int fd, void *buf, size_t n, int write_mode){
    char *p=(char*)buf;
    while(n){
        ssize_t r=write_mode?send(fd,p,n,0):recv(fd,p,n,MSG_WAITALL);
        if(r<=0){ if(r<0&&errno==EINTR) continue; return -1; }
        p+=r; n-=(size_t)r;
    }
    return 0;
}
static int cluster_u32(int fd, uint32_t *v, int write_mode){
    uint32_t x=write_mode?htonl(*v):0;
    if(cluster_io(fd,write_mode?(void*)&x:(void*)v,sizeof(x),write_mode)) return -1;
    if(!write_mode) *v=ntohl(*v);
    return 0;
}
static int cluster_connect_one(const char *spec, ClusterWorker *out){
    char copy[256]; strncpy(copy,spec,sizeof(copy)-1); copy[sizeof(copy)-1]=0;
    char *colon=strrchr(copy,':'); if(!colon||colon==copy||!colon[1]) return -1;
    *colon=0; int port=atoi(colon+1); if(port<1||port>65535) return -1;
    char portbuf[16]; snprintf(portbuf,sizeof(portbuf),"%d",port);
    struct addrinfo hint={0},*ai=NULL; hint.ai_socktype=SOCK_STREAM;
    if(getaddrinfo(copy,portbuf,&hint,&ai)!=0) return -1;
    int fd=-1;
    for(struct addrinfo *p=ai;p;p=p->ai_next){
        fd=socket(p->ai_family,p->ai_socktype,p->ai_protocol);
        if(fd<0) continue;
        if(connect(fd,p->ai_addr,p->ai_addrlen)==0) break;
        close(fd); fd=-1;
    }
    freeaddrinfo(ai); if(fd<0) return -1;
    out->fd=fd;
    size_t hostlen=strlen(copy); if(hostlen>=sizeof(out->host)) hostlen=sizeof(out->host)-1;
    memcpy(out->host,copy,hostlen); out->host[hostlen]=0;
    out->port=port; return 0;
}
static void cluster_close_all(void){
    for(int i=0;i<g_cluster_n;i++) if(g_cluster_workers[i].fd>=0) close(g_cluster_workers[i].fd);
    g_cluster_n=0;
}
static void cluster_init(void){
    const char *list=getenv("CLUSTER_WORKERS"); if(!list||!*list) return;
    char *copy=strdup(list),*save=NULL;
    for(char *tok=strtok_r(copy,",",&save);tok&&g_cluster_n<16;tok=strtok_r(NULL,",",&save)){
        while(*tok==' '||*tok=='\t') tok++;
        if(!cluster_connect_one(tok,&g_cluster_workers[g_cluster_n])) g_cluster_n++;
        else fprintf(stderr,"[CLUSTER] cannot connect to expert worker %s\n",tok);
    }
    free(copy);
    if(g_cluster_n<1){ fprintf(stderr,"[CLUSTER] no expert workers reachable\n"); exit(1); }
    fprintf(stderr,"[CLUSTER] coordinator connected to %d expert worker(s)\n",g_cluster_n);
}
typedef struct { int eid,nr; int *rows; float *weights,*inputs; } ClusterItem;
static int cluster_item(const int *idxs,const float *ws,const int *keff,int K,int S,
                        int eid,ClusterItem *it,int D,const float *x){
    it->eid=eid; it->nr=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++)
        if(idxs[(int64_t)s*K+k]==eid){ it->nr++; break; }
    if(!it->nr) return 0;
    it->rows=malloc((size_t)it->nr*sizeof(int));
    it->weights=malloc((size_t)it->nr*sizeof(float));
    it->inputs=falloc((int64_t)it->nr*D); int r=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++) if(idxs[(int64_t)s*K+k]==eid){
        it->rows[r]=s; it->weights[r]=ws[(int64_t)s*K+k];
        memcpy(it->inputs+(int64_t)r*D,x+(int64_t)s*D,(size_t)D*sizeof(float)); r++; break;
    }
    return 1;
}
static void cluster_item_free(ClusterItem *it){ free(it->rows); free(it->weights); free(it->inputs); memset(it,0,sizeof(*it)); }
static void cluster_moe_batch(Model *m,int layer,float *x,int S,float *out,
                              const int *idxs,const float *ws,const int *keff,int K,
                              const int *uniq,int base,int nb){
    int D=m->c.hidden;
    for(int wi=0;wi<g_cluster_n;wi++){
        ClusterItem items[64]; memset(items,0,sizeof(items)); int n=0;
        for(int j=0;j<nb;j++){
            int eid=uniq[base+j];
            if((eid+layer)%g_cluster_n!=wi) continue;
            if(n<64 && cluster_item(idxs,ws,keff,K,S,eid,&items[n],D,x)) n++;
        }
        if(!n) continue;
        ClusterWorker *w=&g_cluster_workers[wi]; char magic[8]; uint32_t v;
        if(cluster_io(w->fd,(void*)LMA_CLUSTER_MAGIC,8,1)) goto fail;
        v=LMA_CLUSTER_VERSION; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)layer; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)D; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)m->c.moe_inter; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)n; if(cluster_u32(w->fd,&v,1)) goto fail;
        for(int j=0;j<n;j++){
            v=(uint32_t)items[j].eid; if(cluster_u32(w->fd,&v,1)) goto fail;
            v=(uint32_t)items[j].nr; if(cluster_u32(w->fd,&v,1)) goto fail;
            if(cluster_io(w->fd,items[j].inputs,(size_t)items[j].nr*D*sizeof(float),1)) goto fail;
        }
        if(cluster_io(w->fd,magic,8,0)||memcmp(magic,LMA_CLUSTER_MAGIC,8)) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=LMA_CLUSTER_VERSION) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=0) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=(uint32_t)n) goto fail;
        for(int j=0;j<n;j++){
            uint32_t eid,nr;
            if(cluster_u32(w->fd,&eid,0)||cluster_u32(w->fd,&nr,0) ||
               eid!=(uint32_t)items[j].eid || nr!=(uint32_t)items[j].nr) goto fail;
            float *y=falloc((int64_t)nr*D);
            if(cluster_io(w->fd,y,(size_t)nr*D*sizeof(float),0)){ free(y); goto fail; }
            for(uint32_t r=0;r<nr;r++){ float *dst=out+(int64_t)items[j].rows[r]*D;
                float wt=items[j].weights[r]; for(int d=0;d<D;d++) dst[d]+=wt*y[(int64_t)r*D+d]; }
            free(y);
        }
        for(int j=0;j<n;j++) cluster_item_free(&items[j]);
        continue;
fail:
        for(int j=0;j<n;j++) cluster_item_free(&items[j]);
        fprintf(stderr,"[CLUSTER] expert worker %s:%d failed during layer %d batch\n",w->host,w->port,layer);
        exit(1);
    }
}
typedef struct { int eid,nr; float *inputs; } ClusterRequestItem;
/* Mirror expert_load_impl's resolution so the worker can NAME the tensor it can't
 * resolve instead of dying inside a fatal path (st_die_missing / st_read_* exit(1)).
 * Returns 1 and fills `out` with the first missing expert tensor — quantized: a
 * gate/up/down weight or its .qs sidecar; unquantized: the full weight — else 0. */
static int worker_expert_missing_tensor(Model *m,int layer,int eid,char *out,size_t outsz){
    static const char *suf[3]={"gate_proj","up_proj","down_proj"};
    char nm[288], qn[320];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid);
    snprintf(qn,sizeof(qn),"%s.qs",nm);
    if(!st_has(&m->S,qn)){                       /* unquantized: full weights only */
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
            if(!st_has(&m->S,nm)){ snprintf(out,outsz,"%s",nm); return 1; }
        }
        return 0;
    }
    for(int k=0;k<3;k++){                        /* quantized: weight + .qs sidecar */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm);
        if(!st_has(&m->S,nm)||!st_has(&m->S,qn)){ snprintf(out,outsz,"%s",nm); return 1; }
    }
    return 0;
}
static int cluster_worker_run(const char *snap,int port,int ebits,int dbits){
    Model m; memset(&m,0,sizeof(m)); m.ebits=ebits; m.dbits=dbits; load_cfg(&m.c,snap);
    { const char *xd=getenv("LMA_MODEL_DIRS");        /* SPLIT: expert shards spread across N drives */
      st_init_multi(&m.S,snap,(xd&&*xd)?xd:NULL); }
    int nr_layers=m.c.n_layers+1; ESlot *cache=calloc((size_t)nr_layers,sizeof(ESlot));
    for(int i=0;i<nr_layers;i++) cache[i].eid=-1;
    int fd=socket(AF_INET,SOCK_STREAM,0); if(fd<0){perror("cluster worker socket");return 1;}
    int yes=1; setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    struct sockaddr_in addr={0}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=htonl(INADDR_ANY); addr.sin_port=htons((uint16_t)port);
    if(bind(fd,(struct sockaddr*)&addr,sizeof(addr))||listen(fd,4)){perror("cluster worker bind/listen");return 1;}
    fprintf(stderr,"[CLUSTER] expert worker listening on 0.0.0.0:%d (disk-backed, cache=%d/layer)\n",port,nr_layers);
    for(;;){
        int cfd=accept(fd,NULL,NULL); if(cfd<0){if(errno==EINTR)continue;break;}
        for(;;){
            char magic[8]; uint32_t v,layer,D,I,n;
            if(cluster_io(cfd,magic,8,0)) break;
            if(memcmp(magic,LMA_CLUSTER_MAGIC,8)||cluster_u32(cfd,&v,0)||v!=LMA_CLUSTER_VERSION||
               cluster_u32(cfd,&layer,0)||cluster_u32(cfd,&D,0)||cluster_u32(cfd,&I,0)||cluster_u32(cfd,&n,0)||
               D!=(uint32_t)m.c.hidden||I!=(uint32_t)m.c.moe_inter||layer>=(uint32_t)nr_layers||n<1||n>64){
                close(cfd); cfd=-1; break;
            }
            ClusterRequestItem *items=calloc(n,sizeof(*items)); int bad=0;
            for(uint32_t j=0;j<n;j++){
                uint32_t eid,nr;
                if(cluster_u32(cfd,&eid,0)||cluster_u32(cfd,&nr,0)||eid>=(uint32_t)m.c.n_experts||nr<1||nr>65536){bad=1;break;}
                items[j].eid=(int)eid; items[j].nr=(int)nr; items[j].inputs=falloc((int64_t)nr*D);
                if(cluster_io(cfd,items[j].inputs,(size_t)nr*D*sizeof(float),0)){bad=1;break;}
            }
            if(bad){ for(uint32_t j=0;j<n;j++)free(items[j].inputs); free(items); close(cfd); cfd=-1; break; }
            v=LMA_CLUSTER_VERSION; if(cluster_io(cfd,(void*)LMA_CLUSTER_MAGIC,8,1)||cluster_u32(cfd,&v,1)) break;
            v=0; if(cluster_u32(cfd,&v,1)) break; v=n; if(cluster_u32(cfd,&v,1)) break;
            ESlot *slot=&cache[layer];
            for(uint32_t j=0;j<n;j++){
                if(slot->eid!=items[j].eid || !slot->slab){
                    char miss[288];
                    if(worker_expert_missing_tensor(&m,(int)layer,items[j].eid,miss,sizeof(miss))){
                        fprintf(stderr,"[CLUSTER] worker cannot resolve expert tensor %s (layer %d, expert %d)\n",
                                miss,(int)layer,items[j].eid);
                        bad=1; break;
                    }
                    if(expert_load(&m,(int)layer,items[j].eid,slot,0,0)){bad=1;break;}
                }
                int rows=items[j].nr; float *g=falloc((int64_t)rows*I),*u=falloc((int64_t)rows*I),*y=falloc((int64_t)rows*D);
                /* fmt=6: the gate/up input is per-item here (never reused after this
                 * expert), so rotate it in place — the same Q^T x that moe() applies
                 * once per layer via E8_XE. The down-input rotation lives in expert_ffn. */
                if(slot->g.fmt==6) e8_rot_rows(items[j].inputs,rows,D);
                expert_ffn(y,g,u,items[j].inputs,&slot->g,&slot->u,&slot->d,rows,I);
                v=(uint32_t)items[j].eid; if(cluster_u32(cfd,&v,1)){bad=1;free(g);free(u);free(y);break;}
                v=(uint32_t)rows; if(cluster_u32(cfd,&v,1)||cluster_io(cfd,y,(size_t)rows*D*sizeof(float),1)){bad=1;free(g);free(u);free(y);break;}
                free(g);free(u);free(y);
            }
            for(uint32_t j=0;j<n;j++)free(items[j].inputs); free(items);
            if(bad) break;
        }
        if(cfd>=0)close(cfd);
    }
    close(fd); return 0;
}
#endif

#endif /* GLM52_CLUSTER_H */
