#ifndef LMA_DEEPSEEK_V4_H
#define LMA_DEEPSEEK_V4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- config ---- */

#define LMA_V4_MAX_LAYERS 128

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int head_dim;
    int q_lora_rank;
    int qk_rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int sliding_window;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;
    int moe_intermediate_size;
    int num_hash_layers;
    int num_nextn_predict_layers;
    int dspark_block_size;
    int dspark_noise_token_id;
    int dspark_markov_rank;
    int hc_mult;
    int hc_sinkhorn_iters;
    int vocab_size;
    int max_position_embeddings;
    int original_max_position_embeddings;
    int compress_ratio_count;
    int compress_ratios[LMA_V4_MAX_LAYERS];
    float rms_norm_eps;
    float hc_eps;
    float routed_scaling_factor;
    float swiglu_limit;
    float rope_theta;
    float rope_factor;
    float compress_rope_theta;
    int rope_beta_fast;
    int rope_beta_slow;
} LmaDeepSeekV4Config;

int lma_v4_config_parse(LmaDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size);
int lma_v4_config_load(LmaDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size);

/* ---- prompt ---- */

typedef enum {
    LMA_V4_PROMPT_CHAT,
    LMA_V4_PROMPT_THINKING,
    LMA_V4_PROMPT_RAW,
} LmaDeepSeekV4PromptMode;

int lma_v4_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         LmaDeepSeekV4PromptMode mode);

/* ---- experimental public engine API ---- */
/*
 * The API is scoped to the DeepSeek V4 engine and may change while
 * the implementation remains experimental.
 */

typedef struct LmaV4Engine LmaV4Engine;

typedef struct {
    /* Copied by lma_v4_engine_open; caller strings need not outlive the engine. */
    const char *target_model_dir;   /* required */
    uint64_t memory_limit_bytes;    /* 0 => use OS available memory */
    int context_tokens;             /* 0 => 4096 */
    int pin_slots_per_layer;        /* -1 => auto */
    uint64_t repin_interval;        /* 0 => auto */
    int no_dspark;                  /* disable speculative draft/verification */
} LmaV4EngineOpenOptions;

typedef struct {
    uint64_t projected_bytes;
    uint64_t expert_cache_bytes;
    int slots_per_layer;
    int dense_resident;
    int head_resident;
} LmaV4EngineMemorySummary;

int lma_v4_engine_open(LmaV4Engine **engine,
                        const LmaV4EngineOpenOptions *options,
                        char *error, size_t error_size);
/* Undefined if any LmaV4Session created from this engine is still alive.
 * Destroy every session before destroying the engine. */
void lma_v4_engine_destroy(LmaV4Engine *engine);

const LmaDeepSeekV4Config *lma_v4_engine_config(const LmaV4Engine *engine);
void lma_v4_engine_memory_summary(const LmaV4Engine *engine,
                                   LmaV4EngineMemorySummary *summary);

const char *lma_v4_engine_target_model_dir(const LmaV4Engine *engine);

/* ---- experimental public session API ----
 * Session borrows the engine; destroy all sessions before lma_v4_engine_destroy.
 */

typedef struct LmaV4Session LmaV4Session;

typedef struct {
    int max_prompt_tokens;   /* 0 => 512 */
    int max_new_tokens_cap;  /* 0 => 512 */
} LmaV4SessionCreateOptions;

/* Return non-zero to abort generation. Polled between prefill chunks, where
 * the per-token callback cannot fire; a NULL callback keeps prefill
 * uninterruptible as before. */
typedef int (*LmaV4SessionAbortFn)(void *user_data);

/* The numeric channel (SUBMIT logprobs=k, docs/brio.md): raw head scores,
 * vocab_size floats, valid only for the duration of the callback. on_echo
 * fires once per prompt position whose predictor this call computed, with the
 * token that actually stands there; on_scores fires right before the on_token
 * of every generated token, with the scores it was picked from. */
typedef void (*LmaV4SessionScoresFn)(void *user_data, int position, int token,
                                      const float *scores, int vocab);

typedef struct {
    int max_new_tokens;      /* required; clamped by session cap; 0 only with logprobs > 0 */
    int stop_at_sentence;
    int no_dspark;           /* disable speculative draft/verification */
    LmaV4SessionAbortFn should_abort;  /* optional prefill abort poll */
    void *abort_user_data;
    /* SUBMIT logprobs=k / pin=1. logprobs > 0 opens the channel above, turns
     * speculative decoding off for the request (a draft accepted in a block
     * has no scores of its own) and allows max_new_tokens == 0, "read the
     * prompt and stop". pin keeps the prompt-end scores and a snapshot of the
     * attention state, so the prompts that extend this one start from here
     * with their first fresh token's predictor intact. */
    int logprobs;
    int pin;
    LmaV4SessionScoresFn on_echo;
    LmaV4SessionScoresFn on_scores;
    void *scores_user_data;
    /* Optional: byte length of the prompt's stable leading prefix (the
     * rendered system turn). The session snapshots the attention state at
     * that token boundary during this prefill so later conversations that
     * share it start there. 0 = unknown. */
    size_t prefix_bytes;
} LmaV4SessionGenerateOptions;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    int eos_stopped;
    double time_to_first_token_sec;
    double decode_sec;
    uint64_t speculative_drafted;
    uint64_t speculative_accepted;
} LmaV4SessionGenerateStats;

/* Return non-zero to stop generation. */
typedef int (*LmaV4SessionTokenFn)(void *user_data, int token, float logit,
                                    int position, int ordinal);

int lma_v4_session_create(LmaV4Session **session, LmaV4Engine *engine,
                           const LmaV4SessionCreateOptions *options,
                           char *error, size_t error_size);
void lma_v4_session_destroy(LmaV4Session *session);

int lma_v4_session_generate(LmaV4Session *session,
                             const char *prompt, size_t prompt_length,
                             const LmaV4SessionGenerateOptions *options,
                             LmaV4SessionTokenFn on_token, void *user_data,
                             LmaV4SessionGenerateStats *stats,
                             char *error, size_t error_size);

int lma_v4_session_generated_text(const LmaV4Session *session,
                                   char *buffer, size_t buffer_size,
                                   size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif /* LMA_DEEPSEEK_V4_H */
