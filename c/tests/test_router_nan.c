/* Regression for non-finite-logit poisoning of the ROUTER (upstream of sampling).
 *
 * test_logit_nan.c hardened the sampling side. The routing side was not, and it
 * runs first: the three top-K selection loops in moe() start from best=-1 and use
 * it as an index as soon as the scan ends. choice[e] = sigmoidf(logit[e]) + bias,
 * and sigmoidf(NaN) is NaN, so `choice[e] > bv` is false for every e — best stays
 * -1. The old code then did idx[kk]=best; w[kk]=logit[best], which is:
 *
 *   - a heap OOB read of logits_all[s*E - 1] at s==0;
 *   - three OOB writes at index -1 (eusage / eheat / elast);
 *   - a stack buffer underflow on the VLA `unsigned char seen[E]` in FASE B;
 *   - uniq[nu++] = -1 handed to expert_load().
 *
 * Fix under test: moe_router.h maps a failed scan to a deterministic in-range
 * expert and warns once (the shared implementation every engine now uses; its
 * own unit test is tests/test_moe_router.c). This file pins the glm52.c
 * integration -- router_select_topk(), which moe() calls -- so the wiring, not
 * only the helper, is covered. Degrade + diagnose, never index out of range.
 *
 * No model file needed: exercises the guard and the exact scan that trips it. */
#include <assert.h>
#include <math.h>
#define main lma_glm_main_unused
#include "../glm52.c"
#undef main

/* the selection scan as written in moe(), verbatim in shape */
static int scan_best(const float *choice, int E, const int *taken, int ntaken){
    int best=-1; float bv=-1e30f;
    for(int e=0;e<E;e++){
        int tk=0; for(int j=0;j<ntaken;j++) if(taken[j]==e){tk=1;break;}
        if(!tk && choice[e]>bv){bv=choice[e];best=e;}
    }
    return best;
}

int main(void){
    const int E=8;

    /* --- the trigger is real: an all-NaN choice[] leaves best==-1 --- */
    { float choice[8]; for(int e=0;e<E;e++) choice[e]=sigmoidf(NAN)+0.0f;
      for(int e=0;e<E;e++) assert(choice[e]!=choice[e] && "sigmoidf(NaN) is NaN");
      assert(scan_best(choice,E,NULL,0)==-1 && "all-NaN scan selects nothing"); }

    /* one NaN logit is enough to poison a slot once the finite ones are taken */
    { float choice[8]={NAN,NAN,NAN,NAN,NAN,NAN,NAN,2.f};
      int taken[1]={7};
      assert(scan_best(choice,E,taken,1)==-1 && "remaining candidates all NaN"); }

    /* --- the fast scan preserves the old finite-score ordering --- */
    { float choice[8]={0.11f,0.91f,0.37f,0.63f,NAN,0.22f,0.77f,0.48f};
      float fast_choice[8]; memcpy(fast_choice,choice,sizeof choice);
      int old[5], fast[5];
      for(int kk=0;kk<5;kk++){
          int b=scan_best(choice,E,old,kk);
          old[kk]=b>=0?b:kk;            /* the old fallback: the slot's own index */
      }
      router_select_topk(fast_choice,E,5,fast,0);
      for(int kk=0;kk<5;kk++) assert(fast[kk]==old[kk] && "fast top-K ordering changed");
    }

    /* The optimized path keeps the NaN fallback bounded and distinct too. */
    { float choice[8]; for(int e=0;e<E;e++) choice[e]=NAN;
      int idx[4]; router_select_topk(choice,E,4,idx,0);
      for(int kk=0;kk<4;kk++) assert(idx[kk]==kk && "fast fallback must stay deterministic");
    }

    /* --- end to end: the accounting arrays are indexed safely after the guard --- */
    { float choice[8]; for(int e=0;e<E;e++) choice[e]=NAN;
      unsigned char seen[8]; memset(seen,0,sizeof seen);
      int idx[4]; int Ksel=4;
      for(int kk=0;kk<Ksel;kk++){
          idx[kk]=scan_best(choice,E,idx,kk);   /* still -1 here: the unguarded loop */
      }
      router_select_topk(choice,E,Ksel,idx,0);   /* what moe() actually calls */
      for(int kk=0;kk<Ksel;kk++){
          assert(idx[kk]>=0 && idx[kk]<E && "no OOB index reaches eusage/eheat/elast");
          seen[idx[kk]]=1;                 /* would have been seen[-1] before the fix */
      } }

    printf("OK test_router_nan: top-K selection guards best==-1\n");
    return 0;
}
