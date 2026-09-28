/* GLM-5.2 Segment and Edge adapters: the engine side of the layer-range / tokenizer-boundary ABI
 * (segment_runtime.h, edge_runtime.h). Built only with -DCOLI_SEGMENT_ADAPTER / -DCOLI_EDGE_ADAPTER;
 * no CLI path reaches it. Split out of colibri.c so the engine file is the engine.
 *
 * Not a standalone header: colibri.c includes it in place, and the code below uses that
 * file's types and statics (Model, ESlot, QT, now_s, ...). Moving it out changes no
 * behaviour; the translation unit is byte-for-byte the same after preprocessing. */
#ifndef COLIBRI_ADAPTERS_H
#define COLIBRI_ADAPTERS_H

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    KVState *base_kv;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} GlmSegmentEngine;

typedef struct {
    GlmSegmentEngine *engine;
    KVState kv;
    uint32_t context_tokens, position;
} GlmSegmentSession;

static void glm_segment_qt_destroy(QT *tensor) {
    if (!tensor) return;
    /* #826: a file-backed mmap view (mmap_view=1) holds interior pointers into a
     * shard mapping; free() on q8/q4/s would abort. Segment loads are gated to the
     * resident path, but keep this guard so the mmap_view contract holds here too. */
    if (tensor->mmap_view) return;
    free(tensor->qf); free(tensor->q8); free(tensor->q4); free(tensor->s);
    memset(tensor, 0, sizeof(*tensor));
}

static void glm_segment_layer_destroy(Layer *layer) {
    if (!layer) return;
    free(layer->in_ln); free(layer->post_ln);
    free(layer->q_a_ln); free(layer->kv_a_ln);
    glm_segment_qt_destroy(&layer->q_a);
    glm_segment_qt_destroy(&layer->q_b);
    glm_segment_qt_destroy(&layer->kv_a);
    glm_segment_qt_destroy(&layer->kv_b);
    glm_segment_qt_destroy(&layer->o);
    if (layer->sparse) {
        free(layer->router); free(layer->router_bias);
        glm_segment_qt_destroy(&layer->sh_gate);
        glm_segment_qt_destroy(&layer->sh_up);
        glm_segment_qt_destroy(&layer->sh_down);
    } else {
        glm_segment_qt_destroy(&layer->gate_proj);
        glm_segment_qt_destroy(&layer->up_proj);
        glm_segment_qt_destroy(&layer->down_proj);
    }
}

static void glm_segment_eslot_destroy(ESlot *slot) {
    if (!slot) return;
    if (slot->slab || slot->fslab || slot->aslab) {
        if (slot->aslab) {
            /* Segment adapters never build pin arenas, but avoid freeing an
             * interior pointer if a future CPU tier attaches one. */
            slot->slab = NULL; slot->fslab = NULL;
        } else {
            compat_aligned_free(slot->slab);
            free(slot->fslab);
        }
    } else {
        glm_segment_qt_destroy(&slot->g);
        glm_segment_qt_destroy(&slot->u);
        glm_segment_qt_destroy(&slot->d);
    }
    memset(slot, 0, sizeof(*slot));
}

static void glm_segment_kv_destroy(KVState *state, int num_layers) {
    if (!state) return;
    int rows = num_layers + 1;
    if (state->Lc || state->Rc)
        for (int layer = 0; layer < rows; layer++) {
            free(state->Lc ? state->Lc[layer] : NULL);
            free(state->Rc ? state->Rc[layer] : NULL);
        }
    if (state->Lc8 || state->Rc8 || state->Lsc || state->Rsc)
        for (int layer = 0; layer < rows; layer++) {
            free(state->Lc8 ? state->Lc8[layer] : NULL);
            free(state->Rc8 ? state->Rc8[layer] : NULL);
            free(state->Lsc ? state->Lsc[layer] : NULL);
            free(state->Rsc ? state->Rsc[layer] : NULL);
        }
    if (state->Ic)
        for (int layer = 0; layer < num_layers; layer++) free(state->Ic[layer]);
    free(state->Lc); free(state->Rc); free(state->Ic);
    free(state->Lc8); free(state->Rc8);
    free(state->Lsc); free(state->Rsc); free(state->kv_start);
    if (state->disk_fp) fclose(state->disk_fp);
    free(state->disk_buf);
    memset(state, 0, sizeof(*state));
}

static void glm_segment_model_destroy(GlmSegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    int rows = model->c.n_layers + 1;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        glm_segment_layer_destroy(&model->L[layer]);
        if (model->ecache && model->ecache[layer])
            for (int slot = 0; slot < model->ecap; slot++)
                glm_segment_eslot_destroy(&model->ecache[layer][slot]);
        if (model->pin && model->pin[layer])
            for (int slot = 0; slot < model->npin[layer]; slot++)
                glm_segment_eslot_destroy(&model->pin[layer][slot]);
        free(model->ecache ? model->ecache[layer] : NULL);
        free(model->pin ? model->pin[layer] : NULL);
        free(model->eroute ? model->eroute[layer] : NULL);
        free(model->eheat ? model->eheat[layer] : NULL);
        free(model->elast ? model->elast[layer] : NULL);
        free(model->elast_dc ? model->elast_dc[layer] : NULL);
        free(model->elast_pre ? model->elast_pre[layer] : NULL);
    }
    for (size_t slot = 0; slot < sizeof(model->ws) / sizeof(model->ws[0]);
         slot++) glm_segment_eslot_destroy(&model->ws[slot]);
    for (int layer = 0; layer < rows; layer++) {
        free(model->ecache_slot_by_expert
                 ? model->ecache_slot_by_expert[layer] : NULL);
        free(model->pin_slot_by_expert
                 ? model->pin_slot_by_expert[layer] : NULL);
    }
    if (model->has_dsa) {
        for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
             layer++) if (model->c.idx_type[layer]) {
            glm_segment_qt_destroy(&model->ix_wq[layer]);
            glm_segment_qt_destroy(&model->ix_wk[layer]);
            glm_segment_qt_destroy(&model->ix_wp[layer]);
            free(model->ix_knw[layer]); free(model->ix_knb[layer]);
        }
    }
    glm_segment_qt_destroy(&model->embed);
    glm_segment_qt_destroy(&model->lm_head);
    free(model->final_norm); free(model->hlast); free(model->h_all);
    free(model->ix_wq); free(model->ix_wk); free(model->ix_wp);
    free(model->ix_knw); free(model->ix_knb);
    free(model->dsa_sel); free(model->dsa_nsel);
    glm_segment_kv_destroy(engine->base_kv, model->c.n_layers);
    free(engine->base_kv);
    free(model->ecache); free(model->ecn);
    free(model->ecache_slot_by_expert);
    free(model->pin); free(model->npin); free(model->pin_slot_by_expert);
    free(model->eheat); free(model->elast);
    free(model->elast_dc); free(model->elast_pre);
    free(model->eroute); free(model->enr);
    free(model->eusage);
    free(model->kv_dev_L); free(model->kv_dev_R); free(model->kv_dev_valid);
    free(model->ln_dev);
#ifdef COLI_VULKAN
    free(model->vk_kv_valid);
#endif
    free(model->L);
    st_destroy(&model->S);
    model->kv = NULL;
}

static int glm_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid GLM Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment currently supports CPU");
    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    if (options->layer_end > (uint32_t)config.n_layers)
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment range exceeds model");
    int ebits = getenv("GLM_SEGMENT_EBITS")
        ? atoi(getenv("GLM_SEGMENT_EBITS")) : 8;
    int dbits = getenv("GLM_SEGMENT_DBITS")
        ? atoi(getenv("GLM_SEGMENT_DBITS")) : ebits;
    if (ebits < 2 || ebits > 16 || dbits < 2 || dbits > 16)
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment bit widths must be 2..16");
    int sparse_layers = 0;
    for (uint32_t layer = options->layer_begin; layer < options->layer_end;
         layer++) sparse_layers += layer >= (uint32_t)config.first_dense;
    int cap = 8;
    if (options->memory_limit_bytes && sparse_layers) {
        uint64_t values = 3u * (uint64_t)config.moe_inter * config.hidden;
        uint64_t weight_bytes = ebits >= 16 ? values * sizeof(float)
            : ebits >= 5 ? values : ebits >= 4 ? (values + 1u) / 2u
            : ebits == 3 ? (values * 3u + 7u) / 8u
            : (values + 3u) / 4u;
        uint64_t scale_bytes =
            (uint64_t)(2 * config.moe_inter + config.hidden) * sizeof(float);
        uint64_t slot_bytes = weight_bytes + scale_bytes;
        uint64_t slots = slot_bytes
            ? options->memory_limit_bytes / slot_bytes /
              (uint64_t)sparse_layers : 0;
        cap = slots > (uint64_t)config.n_experts
            ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }

    GlmSegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening GLM Segment");
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize GLM Segment lock");
    }
    model_init_range(&engine->model, options->model_dir, cap, ebits, dbits,
                     (int)options->layer_begin, (int)options->layer_end,
                     0, 0, 0, 0);   /* #826: range load stays resident (never a mmap view) */
    engine->base_kv = engine->model.kv;

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "glm");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "glm/mla-rope-dsa-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "glm/e%d-d%d-dsa%d-kvf32/cpu-v1",
             ebits, dbits, engine->model.has_dsa != 0);
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)config.hidden;
    capabilities->max_batch_rows = 512;
    capabilities->max_context_tokens = options->context_tokens;
    capabilities->num_layers = (uint32_t)config.n_layers;
    *engine_impl = engine;
    return 0;
}

static void glm_segment_engine_destroy(void *engine_impl) {
    GlmSegmentEngine *engine = (GlmSegmentEngine *)engine_impl;
    if (!engine) return;
    glm_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static void glm_segment_session_free(GlmSegmentSession *session) {
    if (!session) return;
    if (session->engine)
        glm_segment_kv_destroy(&session->kv,
                               session->engine->model.c.n_layers);
    free(session);
}

static int glm_segment_session_create(
    void *engine_impl, void **session_impl,
    const ColiSegmentSessionOptions *options, char *error, size_t error_size) {
    GlmSegmentEngine *engine = (GlmSegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid GLM Segment session");
    *session_impl = NULL;
    GlmSegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating GLM session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    int rows = engine->model.c.n_layers + 1;
    session->kv.Lc = calloc((size_t)rows, sizeof(*session->kv.Lc));
    session->kv.Rc = calloc((size_t)rows, sizeof(*session->kv.Rc));
    session->kv.kv_start = calloc((size_t)rows,
                                  sizeof(*session->kv.kv_start));
    if (engine->model.has_dsa)
        session->kv.Ic = calloc((size_t)engine->model.c.n_layers,
                                sizeof(*session->kv.Ic));
    if (!session->kv.Lc || !session->kv.Rc || !session->kv.kv_start ||
        (engine->model.has_dsa && !session->kv.Ic)) goto oom;
    session->kv.max_t = (int)options->context_tokens;
    uint64_t state_bytes = 0;
    Cfg *config = &engine->model.c;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        size_t cells;
        if (coli_segment_size_mul(options->context_tokens,
                                  (size_t)config->kv_lora, &cells)) goto oom;
        session->kv.Lc[layer] = calloc(cells, sizeof(float));
        state_bytes += (uint64_t)cells * sizeof(float);
        if (coli_segment_size_mul(options->context_tokens,
                                  (size_t)config->qk_rope, &cells)) goto oom;
        session->kv.Rc[layer] = calloc(cells, sizeof(float));
        state_bytes += (uint64_t)cells * sizeof(float);
        if (engine->model.has_dsa && config->idx_type[layer]) {
            if (coli_segment_size_mul(options->context_tokens,
                                      (size_t)config->index_hd,
                                      &cells)) goto oom;
            session->kv.Ic[layer] = calloc(cells, sizeof(float));
            state_bytes += (uint64_t)cells * sizeof(float);
        }
        if (!session->kv.Lc[layer] || !session->kv.Rc[layer] ||
            (engine->model.has_dsa && config->idx_type[layer] &&
             !session->kv.Ic[layer])) goto oom;
    }
    if (options->memory_limit_bytes &&
        state_bytes > options->memory_limit_bytes) {
        glm_segment_session_free(session);
        return coli_segment_adapter_error(error, error_size,
                                           "GLM session state exceeds memory limit");
    }
    *session_impl = session;
    return 0;

oom:
    glm_segment_session_free(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating GLM state");
}

static void glm_segment_session_destroy(void *session_impl) {
    glm_segment_session_free((GlmSegmentSession *)session_impl);
}

static int glm_segment_session_run(void *session_impl,
                                   const ColiSegmentRunRequest *request,
                                   char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "GLM Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment run cancelled");
    GlmSegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    kv_bind(&engine->model, &session->kv);
    layers_forward_range(&engine->model, (float *)request->output,
                         (int)request->rows, (int)request->position,
                         (int)engine->layer_begin, (int)engine->layer_end);
    kv_bind(&engine->model, engine->base_kv);
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int glm_segment_spans(
    GlmSegmentSession *session, uint32_t position,
    ColiSegmentStateSpan **spans_output, size_t *count_output,
    char *error, size_t error_size) {
    size_t capacity = (size_t)(session->engine->layer_end -
                               session->engine->layer_begin) * 3u;
    ColiSegmentStateSpan *spans = capacity
        ? calloc(capacity, sizeof(*spans)) : NULL;
    if (capacity && !spans)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory describing GLM state");
    Cfg *config = &session->engine->model.c;
    size_t count = 0;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        spans[count++] = (ColiSegmentStateSpan){
            session->kv.Lc[layer],
            (size_t)position * config->kv_lora * sizeof(float)};
        spans[count++] = (ColiSegmentStateSpan){
            session->kv.Rc[layer],
            (size_t)position * config->qk_rope * sizeof(float)};
        if (session->engine->model.has_dsa && config->idx_type[layer])
            spans[count++] = (ColiSegmentStateSpan){
                session->kv.Ic[layer],
                (size_t)position * config->index_hd * sizeof(float)};
    }
    *spans_output = spans; *count_output = count;
    return 0;
}

static int glm_segment_session_snapshot(
    void *session_impl, ColiSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (!session || glm_segment_spans(session, session->position, &spans,
                                      &count, error, error_size))
        return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes)) {
        free(spans);
        return coli_segment_adapter_error(error, error_size,
                                           "GLM snapshot size overflow");
    }
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(
        &header, "glm", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, coli_segment_spans_hash(spans, count));
    int result = coli_segment_stream_write(
        write_fn, write_user_data, &header, sizeof(header), error, error_size);
    if (!result)
        result = coli_segment_spans_write(spans, count, write_fn,
                                          write_user_data, error, error_size);
    free(spans);
    return result;
}

static int glm_segment_session_restore(
    void *session_impl, ColiSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (glm_segment_spans(session, header.position, &spans, &count,
                          error, error_size)) return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes) ||
        coli_segment_snapshot_header_valid(
            &header, "glm", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens,
            payload_bytes, error, error_size)) {
        free(spans); return -1;
    }
    int result = coli_segment_spans_restore(
        spans, count, header.payload_hash, read_fn, read_user_data,
        error, error_size);
    free(spans);
    if (!result) session->position = header.position;
    return result;
}

static const ColiSegmentAdapter glm_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "glm",
    glm_segment_engine_open, glm_segment_engine_destroy,
    glm_segment_session_create, glm_segment_session_destroy,
    glm_segment_session_run, glm_segment_session_snapshot,
    glm_segment_session_restore, {0}
};

int coli_glm_segment_adapter_register(void) {
    return coli_segment_adapter_register(&glm_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

#ifdef COLI_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
    Tok tokenizer;
} GlmEdgeEngine;

static void glm_edge_qt_destroy(QT *tensor) {
    if (!tensor) return;
    /* #826: never free() a file-backed mmap view (interior pointers into a shard
     * mapping). Edge loads embed/lm_head via the resident path today; this guard
     * keeps the mmap_view contract true if that ever changes. */
    if (tensor->mmap_view) return;
    free(tensor->qf); free(tensor->q8); free(tensor->q4); free(tensor->s);
    memset(tensor, 0, sizeof(*tensor));
}

static void glm_edge_engine_destroy(void *engine_impl) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    if (!engine) return;
    glm_edge_qt_destroy(&engine->model.embed);
    glm_edge_qt_destroy(&engine->model.lm_head);
    free(engine->model.final_norm);
    st_destroy(&engine->model.S);
    tok_free(&engine->tokenizer);
    free(engine);
}

static int glm_edge_has_dsa(Model *model) {
    Cfg *config = &model->c;
    int enabled = config->index_topk > 0 && config->index_nh > 0 &&
                  config->index_hd > 0 && config->index_hd <= 256;
    char name[320];
    for (int layer = 0; enabled && layer < config->n_layers; layer++) {
        if (!config->idx_type[layer]) continue;
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.indexer.wq_b.weight", layer);
        if (!st_has(&model->S, name)) enabled = 0;
    }
    if (getenv("DSA") && atoi(getenv("DSA")) == 0) enabled = 0;
    return enabled;
}

static int glm_edge_engine_open(
    void **engine_impl, ColiEdgeCapabilities *capabilities,
    const ColiEdgeEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid GLM Edge open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_EDGE_CAP_CPU))
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge supports CPU only");
    int ebits = getenv("GLM_SEGMENT_EBITS")
        ? atoi(getenv("GLM_SEGMENT_EBITS")) : 8;
    int dbits = getenv("GLM_SEGMENT_DBITS")
        ? atoi(getenv("GLM_SEGMENT_DBITS")) : ebits;
    if (ebits < 2 || ebits > 16 || dbits < 2 || dbits > 16)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge bit widths must be 2..16");
    GlmEdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening GLM Edge");
    Model *model = &engine->model;
    model->ebits = ebits; model->dbits = dbits;
    load_cfg(&model->c, options->model_dir);
    const char *model_dirs = getenv("COLI_MODEL_DIRS");
    st_init_multi(&model->S, options->model_dir,
                  model_dirs && *model_dirs ? model_dirs : NULL);
    int io_bits = dbits >= 8 ? 16 : dbits;
    model->embed = qt_load(model, "model.embed_tokens.weight",
                           model->c.vocab, model->c.hidden, io_bits);
    model->lm_head = qt_load(model, "lm_head.weight",
                             model->c.vocab, model->c.hidden, io_bits);
    model->final_norm = ld(model, "model.norm.weight");
    char tokenizer_path[4096];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    tok_load(&engine->tokenizer, tokenizer_path);

    int has_dsa = glm_edge_has_dsa(model);
    uint64_t resident = (uint64_t)qt_bytes(&model->embed) +
                        (uint64_t)qt_bytes(&model->lm_head) +
                        (uint64_t)model->c.hidden * sizeof(float);
    if (options->memory_limit_bytes && resident > options->memory_limit_bytes) {
        glm_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge exceeds memory limit");
    }
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    capabilities->flags = COLI_EDGE_CAP_TOKENIZE |
                          COLI_EDGE_CAP_DETOKENIZE |
                          COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS |
                          COLI_EDGE_CAP_CPU;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "glm");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "glm/mla-rope-dsa-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "glm/e%d-d%d-dsa%d-kvf32/cpu-v1", ebits, dbits, has_dsa);
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                "glm/cl100k-byte-bpe-v1");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = (uint32_t)model->c.hidden;
    capabilities->vocab_size = (uint32_t)model->c.vocab;
    capabilities->max_batch_rows = 512;
    capabilities->max_context_tokens = UINT32_MAX;
    capabilities->num_layers = (uint32_t)model->c.n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    capabilities->resident_bytes = resident;
    *engine_impl = engine;
    return 0;
}

static int glm_edge_tokenize(
    void *engine_impl, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    return coli_edge_tok_tokenize(&engine->tokenizer, text, text_bytes,
                                  token_ids, token_capacity, token_count,
                                  error, error_size);
}

static int glm_edge_detokenize(
    void *engine_impl, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    return coli_edge_tok_detokenize(&engine->tokenizer, token_ids, token_count,
                                    text, text_capacity, text_bytes,
                                    error, error_size);
}

static int glm_edge_embed(void *engine_impl,
                          const ColiEdgeEmbedRequest *request,
                          char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    int hidden = engine->model.c.hidden;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        int token = request->token_ids[row];
        if (token < 0 || token >= engine->model.c.vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "GLM token ID is out of range");
        embed_row(&engine->model, token, output + (size_t)row * hidden);
    }
    return 0;
}

static int glm_edge_select(void *engine_impl,
                           const ColiEdgeSelectRequest *request,
                           char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    float *logits = falloc(config->vocab);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge selection cancelled");
        }
        rmsnorm(normalized, input + (size_t)row * config->hidden,
                engine->model.final_norm, config->hidden, config->eps);
        matmul_qt(logits, normalized, &engine->model.lm_head, 1);
        if (coli_edge_argmax(logits, (uint32_t)config->vocab,
                            &request->token_ids[row],
                            request->scores ? &request->scores[row] : NULL)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge head failed");
        }
    }
    free(logits); free(normalized);
    return 0;
}

static int glm_edge_logits(void *engine_impl,
                           const ColiEdgeLogitsRequest *request,
                           char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge logits cancelled");
        }
        rmsnorm(normalized, input + (size_t)row * config->hidden,
                engine->model.final_norm, config->hidden, config->eps);
        matmul_qt(request->logits + (size_t)row * config->vocab,
                  normalized, &engine->model.lm_head, 1);
    }
    free(normalized);
    return 0;
}

static const ColiEdgeAdapter glm_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "glm",
    glm_edge_engine_open, glm_edge_engine_destroy,
    glm_edge_tokenize, glm_edge_detokenize,
    glm_edge_embed, glm_edge_select, glm_edge_logits, {0}
};

int coli_glm_edge_adapter_register(void) {
    return coli_edge_adapter_register(&glm_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */

#endif /* COLIBRI_ADAPTERS_H */
