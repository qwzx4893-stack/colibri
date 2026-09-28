/* expert_host_ensure() must not reload an expert whose weights are already on the host.
 *
 * On a CUDA build it used to test `slab` alone. An unquantized model (no .qs container)
 * never has a slab -- qt_from_disk allocates each tensor's views itself -- and MMAP=1 points
 * the views into the mapping, so every use of a RESIDENT expert re-read and re-quantized it
 * from disk: 2.1 GB instead of 88 MB fetched and 2.2 tok/s instead of 41 on the synthetic
 * bench model (GPU never involved). expert_host_present() is the predicate both need.
 *
 * No model, no GPU: exercises the predicate on a hand-built slot. */
#include <assert.h>
#include <string.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

int main(void) {
    ESlot s;
    memset(&s, 0, sizeof s);
    assert(!expert_host_present(&s) && "a released / never-loaded slot has nothing on the host");

    static unsigned char bytes[16];
    s.slab = bytes;
    assert(expert_host_present(&s) && "slab-backed (quantized container)");
    s.slab = NULL;

    s.g.q4 = bytes;
    assert(expert_host_present(&s) && "views without a slab (unquantized model path)");
    s.g.q4 = NULL; s.g.q8 = (int8_t *)bytes;
    assert(expert_host_present(&s) && "int8 views without a slab (mmap-backed)");
    s.g.q8 = NULL; s.g.qf = (float *)bytes;
    assert(expert_host_present(&s) && "f32 views");
    s.g.qf = NULL;
    assert(!expert_host_present(&s) && "views NULLed by expert_host_release: reload is needed");

    printf("OK test_expert_host_present\n");
    return 0;
}
