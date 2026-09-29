#ifndef LMA_EDGE_ADAPTERS_H
#define LMA_EDGE_ADAPTERS_H

/* Explicit registration keeps the ordinary LM Atomic CLI initialization and
 * Windows/MSVC builds independent from the distributed Edge runtime. */

#ifdef __cplusplus
extern "C" {
#endif

int lma_glm_edge_adapter_register(void);
int lma_glm53_edge_adapter_register(void);
int lma_inkling_edge_adapter_register(void);
int lma_kimi_edge_adapter_register(void);
int lma_olmoe_edge_adapter_register(void);
int lma_qwen36_edge_adapter_register(void);
int lma_qwen38_edge_adapter_register(void);
int lma_deepseek_v4_edge_adapter_register(void);

#ifdef __cplusplus
}
#endif

#endif /* LMA_EDGE_ADAPTERS_H */
