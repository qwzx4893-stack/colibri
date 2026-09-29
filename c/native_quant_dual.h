#ifndef LMA_NATIVE_QUANT_DUAL_H
#define LMA_NATIVE_QUANT_DUAL_H

#include "tensor.h"

int lma_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const LmaTensorView *weight_a,
                             const LmaTensorView *weight_b,
                             const float *input);
int lma_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const LmaTensorView *weight_a,
                             const LmaTensorView *weight_b,
                             const float *input);

#endif
