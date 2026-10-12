import io
p = r"C:\Users\STS\projects\colibri\c\olmoe.c"
s = open(p, encoding="utf-8").read()
gpu = open(r"C:\Users\STS\lma_ab\gpu_block.c.txt", encoding="utf-8").read()

# 1. Model fields
anchor = "    /* LM ATOMIC Physical Giant Emulator ONLY (see g_giant_stick above). */\n    int *giant_prev_idx;"
fields = """    /* LM ATOMIC GPU decode path (olmoe_cuda.cu, loaded from lma_olmoe_cuda.dll). */
    int gpu_on;                      /* 1 = S==1 decode runs on the GPU */
    int gpu_max_t;                   /* device KV capacity in positions; beyond it decode falls back to the CPU */
    int gpu_kv_len;                  /* device KV holds positions [0, gpu_kv_len) */
    int gpu_n_slots, gpu_all_resident;
    int *gpu_slot_of;                /* [L*E] host mirror of the device (layer,expert)->slot table, -1 = absent */
    int *gpu_slot_owner;             /* [n_slots] layer*E+expert, -1 = free */
    uint64_t *gpu_slot_use, gpu_clock;
    float *gpu_kv_stage;             /* [L][2][D] */
    uint64_t gpu_tokens, gpu_miss, gpu_fetch_syncs;
    double gpu_s;                    /* wall seconds spent in gpu_decode */
    float gpu_max_dlogit; uint64_t gpu_verify_n, gpu_verify_agree;   /* LMA_GPU_VERIFY statistics */
"""
assert anchor in s
s = s.replace(anchor, fields + anchor, 1)

# 2. GPU block before step()
marker = "static float *step(Model *m, const int *ids, int S, int pos_base) {"
assert marker in s
s = s.replace(marker, gpu + marker, 1)

# 3. step() fast path
old = """    Cfg *c = &m->c; int D = c->hidden;
    if (g_pilot && m->token_count > 0) {
        /* Flush stale prefetch requests"""
new = """    Cfg *c = &m->c; int D = c->hidden;
    if (m->gpu_on && S == 1 && !g_gpu_verify) {
        float *lg = falloc(c->vocab);
        if (gpu_decode(m, ids[0], pos_base, lg, 1)) {
            m->token_count += 1; m->freq_token_count += 1;
            m->kv_len = pos_base + 1;
            kv_prefix_record(&m->kvp, ids, pos_base, 1);
            g_prof_forwards += 1;
            return lg;
        }
        free(lg);
        m->gpu_on = 0;   /* circuit breaker: this session continues on the CPU */
        fprintf(stderr, "[GPU] decode failed at position %d: falling back to the CPU for the rest of the session\\n", pos_base);
    }
    if (m->gpu_on && pos_base < m->gpu_kv_len) m->gpu_kv_len = pos_base;   /* CPU rewrites these rows */
    if (g_pilot && m->token_count > 0) {
        /* Flush stale prefetch requests"""
assert old in s
s = s.replace(old, new, 1)

# 4. verify mode
old = """    g_prof_forwards += 1;   /* forward passes, not positions: a prefill of S rows is one */
    free(x); free(last);
    return logit;
}"""
new = """    g_prof_forwards += 1;   /* forward passes, not positions: a prefill of S rows is one */
    free(x); free(last);
    if (m->gpu_on && g_gpu_verify && S == 1) {
        /* CPU is the reference (its logits are returned); the GPU runs the same token on its own KV */
        float *lg = falloc(c->vocab);
        if (gpu_decode(m, ids[0], pos_base, lg, 0)) {
            float mx = 0.f; int ac = 0, ag = 0;
            for (int i = 0; i < c->vocab; i++) {
                float d = fabsf(lg[i] - logit[i]); if (d > mx) mx = d;
                if (lg[i] > lg[ag]) ag = i;
                if (logit[i] > logit[ac]) ac = i;
            }
            if (mx > m->gpu_max_dlogit) m->gpu_max_dlogit = mx;
            m->gpu_verify_n++; if (ag == ac) m->gpu_verify_agree++;
        } else { m->gpu_on = 0; fprintf(stderr, "[GPU] verify decode failed; GPU disabled\\n"); }
        free(lg);
    }
    return logit;
}"""
assert old in s
s = s.replace(old, new, 1)

# 5. dlfcn include for non-Windows
old = "#ifdef _WIN32\n#include <windows.h>\n#define sleep_ms(ms) Sleep(ms)"
assert old in s
s = s.replace(old, "#ifdef _WIN32\n#include <windows.h>\n#define sleep_ms(ms) Sleep(ms)", 1)
if "#include <dlfcn.h>" not in s:
    s = s.replace("#include <pthread.h>\n", "#include <pthread.h>\n#ifndef _WIN32\n#include <dlfcn.h>\n#endif\n", 1)

open(p, "w", encoding="utf-8").write(s)
print("patched")
