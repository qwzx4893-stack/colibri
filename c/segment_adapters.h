#ifndef LMA_SEGMENT_ADAPTERS_H
#define LMA_SEGMENT_ADAPTERS_H

/*
 * Explicit registration entry points for LM Atomic's built-in model adapters.
 *
 * Each engine owns its implementation and is normally linked as a separate
 * executable/object.  A Segment consumer links the engines it wants and calls
 * the matching functions before the first adapter lookup.  The ordinary
 * LM Atomic CLI/server paths do not call these functions, so adding an adapter
 * cannot change standalone inference or initialization order.
 */

#ifdef __cplusplus
extern "C" {
#endif

int lma_glm_segment_adapter_register(void);
int lma_glm53_segment_adapter_register(void);
int lma_inkling_segment_adapter_register(void);
int lma_kimi_segment_adapter_register(void);
int lma_olmoe_segment_adapter_register(void);
int lma_qwen36_segment_adapter_register(void);
int lma_qwen38_segment_adapter_register(void);
int lma_deepseek_v4_segment_adapter_register(void);

#ifdef __cplusplus
}
#endif

#endif /* LMA_SEGMENT_ADAPTERS_H */
