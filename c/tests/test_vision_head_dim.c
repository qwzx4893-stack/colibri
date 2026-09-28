/* vision_tower.h must refuse a head_dim larger than its rotary scratch instead of
 * silently rotating only the first COLI_VISION_MAX_HEAD_DIM channels.
 *
 * The rotation used to clamp `limit = hd < 512 ? hd : 512`: a checkpoint with a
 * wider head produced a tower whose rotary embedding was quietly truncated --
 * plausible-looking image features, wrong. Refusing is the fail-loud choice, and
 * it is decided before any tensor is touched, so a zeroed tower is enough. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../vision_tower.h"

int main(void) {
    ColiVisionTower tower;
    memset(&tower, 0, sizeof tower);
    float pixel = 0.f, out[8];

    /* head_dim > limit: refused up front (hidden == heads * head_dim, head_dim % 4 == 0) */
    tower.config.heads = 1;
    tower.config.head_dim = COLI_VISION_MAX_HEAD_DIM + 4;
    tower.config.hidden = tower.config.head_dim;
    tower.config.merge = 1;
    assert(coli_vision_forward(out, &tower, &pixel, 1, 1) == -1);

    printf("OK test_vision_head_dim: oversized head_dim is refused, not truncated\n");
    return 0;
}
