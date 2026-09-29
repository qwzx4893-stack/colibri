#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    assert(lma_glm_segment_adapter_register() == 0);
    assert(lma_glm53_segment_adapter_register() == 0);
    assert(lma_inkling_segment_adapter_register() == 0);
    assert(lma_kimi_segment_adapter_register() == 0);
    assert(lma_olmoe_segment_adapter_register() == 0);
    assert(lma_qwen36_segment_adapter_register() == 0);
    assert(lma_qwen38_segment_adapter_register() == 0);
    assert(lma_deepseek_v4_segment_adapter_register() == 0);
    assert(lma_segment_adapter_count() == 8);

    static const char *expected[] = {
        "glm", "glm53", "inkling", "kimi", "olmoe", "qwen36", "qwen38", "deepseek_v4"
    };
    for (size_t item = 0; item < sizeof(expected) / sizeof(expected[0]); item++)
        assert(lma_segment_adapter_lookup(expected[item]) != NULL);
    puts("every real Segment adapter registers: ok");
    return 0;
}
