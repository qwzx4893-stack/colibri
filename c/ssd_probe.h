/* Storage-bandwidth probe (#379): is the model volume fast enough that the OS page cache beats
 * the engine's own expert LRU? Decision layer (cache grammar, cold-tile steering, veto) is pure
 * portable C, tested by tests/test_ssd_probe.c via -DCOLI_SSD_PROBE_TEST.
 *
 * Not a standalone header: colibri.c includes it in place, and the code below uses that
 * file's types and statics (Model, ESlot, QT, now_s, ...). Moving it out changes no
 * behaviour; the translation unit is byte-for-byte the same after preprocessing. */
#ifndef COLIBRI_SSD_PROBE_H
#define COLIBRI_SSD_PROBE_H

/* ---- #379: honest Metal-cache storage probe (S1) --------------------------
 * Metal's per-buffer residency cost makes the engine's own expert LRU cache
 * anti-productive on fast NVMe: the OS page cache already streams cold
 * experts cheaper than Metal can re-pin them (maintainer-endorsed root cause,
 * issue #379). "Fast" has to be MEASURED, not assumed -- buffered reads lie
 * (page-cache hits read 23-97 GB/s on the reference box; the real F_NOCACHE
 * number was 14 GB/s), so this probes the actual model volume the same way
 * compat_open_direct()/iobench.c's __APPLE__ branch do: open + F_NOCACHE,
 * random 16K-aligned pread, no readahead. F_NOCACHE alone is NOT enough: it
 * bypasses the cache for NEW pages but still serves already-resident ones, so
 * a warm shard measured 22.1 GB/s of RAM and latched it as "disk". The probe
 * therefore steers its reads to ranges mincore reports cold, and refuses to
 * conclude anything when the file offers too few cold bytes (veto below).
 * Gating: the whole block builds only for the one consumer (COLI_METAL on
 * __APPLE__), so the default build carries no orphaned probe code; tests
 * define COLI_SSD_PROBE_TEST to compile it anywhere -- the decision layer
 * (grammar parser, cold-tile steering, veto) is pure portable C by design,
 * exercised by tests/test_ssd_probe.c on every platform. */
#if (defined(COLI_METAL) && defined(__APPLE__)) || defined(COLI_SSD_PROBE_TEST)
#define COLI_SSD_PROBE_BLK   (4u*1024*1024)     /* 4 MB reads, 16K-aligned offsets */
#define COLI_SSD_PROBE_MAXB  (300*1024*1024)    /* "a few hundred MB", capped on fast media */
#define COLI_SSD_PROBE_SECS  0.35               /* wall-clock budget, well under the ~1s target */
#define COLI_SSD_PROBE_COLD_FLOOR (64ll*1024*1024) /* fewer cold bytes than this = contaminated */
#define COLI_SSD_CACHE_MAX   64                 /* longest well-formed .coli_ssd, in bytes */

/* Why a measurement was refused (contaminated-class: nothing cached, one
 * honest stderr line naming the ACTUAL condition, retry on a later start).
 * The reasons differ in what the user should do about them -- "wait for a
 * cold start" vs "finish the download" vs "nothing, this shard can never
 * host a trustworthy probe" -- so the message must not lump them. */
enum {
    COLI_SSD_VETO_NONE = 0,
    COLI_SSD_VETO_RESIDENT,                     /* too few cold pages: page cache would answer */
    COLI_SSD_VETO_SMALL,                        /* shard offers < COLD_FLOOR of whole windows, ever */
    COLI_SSD_VETO_SPARSE                        /* under-allocated: holes read as VFS zero-fill */
};

/* Under-allocation gate (#386 round 2, F1): steering prefers never-resident
 * pages, and in a sparse or still-downloading file those are HOLES -- pread
 * of a hole is VFS zero-fill at RAM speed (a 10 GB shard with 100 MB of real
 * data measured 73 GB/s and latched it as disk bandwidth). st_blocks*512 is
 * what actually sits on disk; require it to cover the file size within 12.5%
 * slack (HFS+/APFS metadata rounding, tail blocks). Same conservative
 * polarity as the residency veto: not fully allocated -> not measurable. */
static int coli_ssd_probe_underallocated(long long allocated, long long size){
    return allocated < size - size/8;
}

/* digits [ "." digits ] -- returns chars consumed, 0 = no match. The ONLY
 * number shape the cache grammar admits (see coli_ssd_cache_parse). */
static size_t coli_ssd_scan_number(const char *p, size_t len){
    size_t i=0;
    while(i<len && p[i]>='0' && p[i]<='9') i++;
    if(!i) return 0;
    if(i<len && p[i]=='.'){
        size_t j=i+1;
        while(j<len && p[j]>='0' && p[j]<='9') j++;
        if(j==i+1) return 0;                     /* "5." is garbage */
        i=j;
    }
    return i;
}

/* Strict .coli_ssd grammar -- mirrored byte-for-byte by resource_plan.py's
 * parse_ssd_cache() (keep the two in lockstep; test_ssd_probe.c and
 * test_resource_plan.py chew the same vector file,
 * tests/fixtures/ssd_cache_vectors.txt):
 *   v2:     "v2 <gbs> <st_dev>"   single spaces, at most one trailing \n
 *   legacy: "<gbs>"               the pre-fix format, never trusted (re-probed)
 * where <gbs> = digits["."digits], 0 < gbs < 1000, and <st_dev> = 1..20
 * digits fitting unsigned 64-bit. Total length <= COLI_SSD_CACHE_MAX bytes,
 * no NULs, nothing else. Everything outside the grammar is garbage and
 * re-probes: strtod permissiveness ("inf", "nan", "0x8", "1e99", leading
 * whitespace/signs) is deliberately out -- it let hand-edited or corrupt
 * caches masquerade as measurements.
 * Returns 2 = v2 (gbs+stdev filled), 1 = legacy (gbs filled), 0 = garbage. */
static int coli_ssd_cache_parse(const char *buf, size_t len, double *gbs_out,
                                unsigned long long *stdev_out){
    if(!len || len>COLI_SSD_CACHE_MAX || memchr(buf,0,len)) return 0;
    if(buf[len-1]=='\n') len--;                  /* at most one trailing newline */
    int v2 = len>=3 && !memcmp(buf,"v2 ",3);
    const char *p = buf + (v2?3:0);
    size_t left = len - (v2?3:0);
    size_t n = coli_ssd_scan_number(p,left);
    if(!n) return 0;
    char num[COLI_SSD_CACHE_MAX+1];
    memcpy(num,p,n); num[n]=0;
    double gbs=strtod(num,NULL);                 /* shape already vetted above */
    if(!(gbs>0 && gbs<1000)) return 0;
    p+=n; left-=n;
    if(!v2){
        if(left) return 0;
        *gbs_out=gbs;
        return 1;
    }
    if(!left || *p!=' ') return 0;
    p++; left--;
    size_t d=0;
    while(d<left && p[d]>='0' && p[d]<='9') d++;
    if(!d || d!=left || d>20) return 0;
    char dev[21];
    memcpy(dev,p,d); dev[d]=0;
    errno=0;
    unsigned long long sd=strtoull(dev,NULL,10);
    if(errno) return 0;                          /* 20 nines overflows u64: garbage */
    *gbs_out=gbs; *stdev_out=sd;
    return 2;
}

/* The single writer of the grammar above: "v2 %.3f %llu\n". %.3f of a value
 * in (0,1000) is always digits.digits, so what this writes always re-parses
 * -- the round-trip is pinned by test_ssd_probe.c. */
static int coli_ssd_cache_format(char *buf, size_t bufsz, double gbs,
                                 unsigned long long stdev){
    int n=snprintf(buf,bufsz,"v2 %.3f %llu\n",gbs,stdev);
    return (n>0 && (size_t)n<bufsz) ? n : -1;
}

/* Probe steering (#386 fix round): pure, so tests inject residency vectors --
 * no mincore, no files. vec has one byte per page_sz-sized page, mincore
 * convention (bit 0 set = resident in RAM). The file is tiled into
 * non-overlapping COLI_SSD_PROBE_BLK windows; a window qualifies only when
 * EVERY page in it is non-resident, because one warm page lets the page cache
 * serve part of a "disk" read (a fully-resident shard measured 22.1 GB/s
 * through F_NOCACHE and latched the fast-SSD default on unknown hardware).
 * Emits qualifying window start offsets into tiles[] (caller sizes it at
 * file_size/BLK+1 entries) and returns the count; *cold_bytes_out gets
 * count*BLK, the bytes the probe can trust. */
static size_t coli_ssd_probe_cold_tiles(const unsigned char *vec, size_t npages,
                                        size_t page_sz, long long *tiles,
                                        long long *cold_bytes_out){
    size_t count=0;
    if(page_sz && page_sz<=COLI_SSD_PROBE_BLK){
        size_t per=(COLI_SSD_PROBE_BLK+page_sz-1)/page_sz;
        for(size_t t=0; t+per<=npages; t+=per){
            size_t i=0;
            while(i<per && !(vec[t+i]&1)) i++;
            if(i==per) tiles[count++]=(long long)t*(long long)page_sz;
        }
    }
    if(cold_bytes_out) *cold_bytes_out=(long long)count*COLI_SSD_PROBE_BLK;
    return count;
}

/* Contamination veto (#386 fix round): with fewer than COLI_SSD_PROBE_COLD_FLOOR
 * verified-cold bytes to read, any measurement would be mostly RAM. An
 * untrustworthy measurement behaves as SLOW: no number, no cache write, the
 * conservative default holds, and the probe retries on a colder startup. */
static int coli_ssd_probe_vetoed(long long cold_bytes){
    return cold_bytes < COLI_SSD_PROBE_COLD_FLOOR;
}

#ifdef __APPLE__
/* Path of the LARGEST *.safetensors shard in snap_dir into out, or out[0]=0
 * if none. Any shard's bandwidth is representative, but the biggest one
 * offers the most cold 4 MB windows -- a stray small shard picked by readdir
 * order could starve the probe below the 64 MB floor forever (#386 round 2,
 * F4) even though a full-size neighbor sits right next to it. */
static void coli_ssd_probe_pick_shard(const char *snap_dir, char *out, size_t outsz){
    out[0]=0;
    DIR *d=opendir(snap_dir);
    if(!d) return;
    struct dirent *e;
    long long best=-1;
    while((e=readdir(d))){
        const char *dot=strrchr(e->d_name,'.');
        if(!dot || strcmp(dot,".safetensors")) continue;
        char cand[1280];
        snprintf(cand,sizeof(cand),"%s/%s",snap_dir,e->d_name);
        struct stat st;
        if(stat(cand,&st)==0 && (long long)st.st_size>best){
            best=(long long)st.st_size;
            snprintf(out,outsz,"%s",cand);
        }
    }
    closedir(d);
}

/* F_NOCACHE random-read probe on one real shard, steered to verified-cold
 * ranges. Returns measured GB/s, or -1 if the probe could not run (no shard,
 * open/map failure, file too small) or was vetoed as contaminated -- callers
 * treat -1 as "not fast", never as an error to surface; *veto_out gets the
 * COLI_SSD_VETO_* reason so the caller can say honestly why and skip the
 * cache write. Veto order: SPARSE (holes would measure as zero-fill) before
 * SMALL (can never offer the floor) before RESIDENT (cold today, maybe).
 * The mmap is address space only (nothing is faulted in): it exists so
 * mincore can fill the residency vector. */
static double coli_ssd_probe_raw(const char *snap_dir, int *veto_out){
    if(veto_out) *veto_out=COLI_SSD_VETO_NONE;
    char path[1280];
    coli_ssd_probe_pick_shard(snap_dir,path,sizeof(path));
    if(!path[0]) return -1;
    int fd=open(path,O_RDONLY);
    if(fd<0) return -1;
    fcntl(fd,F_NOCACHE,1);
    fcntl(fd,F_RDAHEAD,0);                       /* random access, not sequential streaming */
    struct stat st;
    if(fstat(fd,&st)!=0 || st.st_size<(off_t)(COLI_SSD_PROBE_BLK*2)){ close(fd); return -1; }
    if(coli_ssd_probe_underallocated((long long)st.st_blocks*512,(long long)st.st_size)){
        if(veto_out) *veto_out=COLI_SSD_VETO_SPARSE;
        close(fd);
        return -1;
    }
    if(((long long)st.st_size/COLI_SSD_PROBE_BLK)*COLI_SSD_PROBE_BLK<COLI_SSD_PROBE_COLD_FLOOR){
        if(veto_out) *veto_out=COLI_SSD_VETO_SMALL;   /* even fully cold it cannot reach the floor */
        close(fd);
        return -1;
    }
    size_t page_sz=(size_t)getpagesize();
    size_t npages=((size_t)st.st_size+page_sz-1)/page_sz;
    unsigned char *vec=malloc(npages);
    long long *tiles=malloc(((size_t)st.st_size/COLI_SSD_PROBE_BLK+1)*sizeof *tiles);
    void *map=(vec&&tiles)?mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0):MAP_FAILED;
    double gbs=-1;
    if(map!=MAP_FAILED && mincore(map,(size_t)st.st_size,(char*)vec)==0){
        long long cold_bytes=0;
        size_t ntiles=coli_ssd_probe_cold_tiles(vec,npages,page_sz,tiles,&cold_bytes);
        if(ntiles==0 || coli_ssd_probe_vetoed(cold_bytes)){
            /* ntiles==0 is implied by the floor today, but the modulo below
             * must NEVER be reachable with 0 on its own merits (#386 r2, F7) */
            if(veto_out) *veto_out=COLI_SSD_VETO_RESIDENT;
        }else{
            void *buf=NULL;
            if(posix_memalign(&buf,16384,COLI_SSD_PROBE_BLK)!=0) buf=NULL;
            if(buf){
                unsigned seed=0xC0117125u;
                double t0=now_s(); long long total=0;
                while(now_s()-t0<COLI_SSD_PROBE_SECS && total<COLI_SSD_PROBE_MAXB){
                    long long off=tiles[(size_t)rand_r(&seed)%ntiles];
                    ssize_t got=pread(fd,buf,COLI_SSD_PROBE_BLK,off);
                    if(got<=0) break;
                    total+=got;
                }
                double dt=now_s()-t0;
                if(total>0 && dt>0) gbs=(double)total/1e9/dt;
                free(buf);
            }
        }
    }
    if(map!=MAP_FAILED) munmap(map,(size_t)st.st_size);
    free(tiles); free(vec);
    close(fd);
    return gbs;
}

/* Cache the measurement in <snap>/.coli_ssd so every startup after the first
 * reads a file instead of re-probing. v2 records the volume's identity
 * ("v2 <gbs> <st_dev>"): a cache is trusted ONLY when it parses under the
 * strict grammar above AND its st_dev matches the model dir's current device
 * -- a dir rsync'd to another volume (external drive, RAM disk) carries a
 * measurement of the WRONG hardware and must re-probe. Legacy bare-number
 * caches (written before steering existed, so possibly warm-contaminated)
 * re-probe once and upgrade to v2. A vetoed run writes nothing and says why
 * on stderr; the probe retries naturally on the next, colder, startup. */
static double coli_ssd_probe_cached(const char *snap_dir){
    char cpath[1290];
    snprintf(cpath,sizeof(cpath),"%s/.coli_ssd",snap_dir);
    struct stat dst;
    unsigned long long cur_dev = stat(snap_dir,&dst)==0 ? (unsigned long long)dst.st_dev : 0;
    FILE *f=fopen(cpath,"rb");                   /* exact bytes: never let a text-mode
                                                  * layer eat the \r the grammar rejects */
    if(f){
        char buf[COLI_SSD_CACHE_MAX+2];
        size_t n=fread(buf,1,sizeof(buf),f);
        fclose(f);
        double cached=0; unsigned long long sdev=0;
        if(coli_ssd_cache_parse(buf,n,&cached,&sdev)==2 && sdev==cur_dev)
            return cached;
        /* v2 from another volume, legacy, or garbage: fall through, re-measure */
    }
    int veto=COLI_SSD_VETO_NONE;
    double gbs=coli_ssd_probe_raw(snap_dir,&veto);
    if(veto!=COLI_SSD_VETO_NONE){
        /* one honest line per deferral, naming the condition that actually
         * fired (#386 r2, F4): "resident" advice (wait for a cold start) is
         * wrong advice for a partial download or a permanently tiny shard. */
        const char *why =
            veto==COLI_SSD_VETO_SPARSE ? "the shard is not fully allocated on disk (partial "
                                         "download?), holes would measure as RAM-speed zero-fill" :
            veto==COLI_SSD_VETO_SMALL  ? "the largest shard offers under 64 MB of probe windows, "
                                         "too small for a trustworthy measurement" :
                                         "the shard is already page-cache resident (<64 MB cold), "
                                         "a read now would measure RAM, not the disk";
        fprintf(stderr,"METAL: storage probe deferred -- %s; keeping the conservative cache "
                "default, will re-check on a later start\n",why);
        return -1;
    }
    if(gbs<0) return -1;                         /* could not run at all: no shard, open failure */
    if(!(gbs>0 && gbs<1000)){
        /* a number the cache grammar itself would refuse (0, >=1000 GB/s,
         * non-finite) is no more trustworthy for THIS run than for the next
         * one (#386 r2, F6): treat it as contaminated, not as "fast". */
        fprintf(stderr,"METAL: storage probe measured an implausible %g GB/s -- ignoring the "
                "measurement, keeping the conservative cache default\n",gbs);
        return -1;
    }
    char line[COLI_SSD_CACHE_MAX+1];
    if(coli_ssd_cache_format(line,sizeof(line),gbs,cur_dev)>0){
        /* mkstemp (unique name, O_EXCL) + rename: the model dir is shared (a
         * concurrent serve + run pair may both probe a virgin dir) so a torn
         * write must never survive, and a fixed, predictable temp name could
         * be pre-planted as a symlink -- mkstemp cannot follow one. Two
         * racing probers both measure under mutual contention: a LOW reading,
         * the safe direction (the platform default just stays off). */
        char tpath[1310];
        snprintf(tpath,sizeof(tpath),"%s.XXXXXX",cpath);
        int cache_errno=0;
        int tfd=mkstemp(tpath);
        if(tfd<0) cache_errno=errno;
        else{
            size_t len=strlen(line);
            mode_t um=umask(0); umask(um);       /* read it back; umask() has no getter */
            int ok = write(tfd,line,len)==(ssize_t)len;
            ok = fchmod(tfd,(mode_t)(0644&~um))==0 && ok;  /* mkstemp forces 0600; honor the umask instead */
            if(!ok) cache_errno=errno;
            close(tfd);
            if(ok && rename(tpath,cpath)!=0){ cache_errno=errno; ok=0; }
            if(!ok) unlink(tpath);
        }
        if(cache_errno)
            /* the measurement still governs THIS run; only persistence failed
             * (read-only model dir, quota, ...) -- say so once (#386 r2, F5) */
            fprintf(stderr,"METAL: could not cache the storage probe in %s (%s) -- "
                    "will re-measure on every start until it is writable\n",
                    cpath,strerror(cache_errno));
    }
    return gbs;
}
#endif /* __APPLE__ */
#endif /* (COLI_METAL && __APPLE__) || COLI_SSD_PROBE_TEST */

#endif /* COLIBRI_SSD_PROBE_H */
