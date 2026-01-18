// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <string.h>
#endif
#include "include/gemmini_testutils.h"

// Input types are defined by Gemmini params: elem_t is usually int8_t
// Accumulator type is acc_t (usually int32_t)

// Define matrix dimensions
#define M  37
#define N  29
#define K  33

static elem_t A[M*K];
static elem_t B[K*N];
static acc_t  C[M*N];
static acc_t  C_ref[M*N];

static inline uint64_t rdcycle(void) {
    uint64_t x;
    asm volatile ("rdcycle %0" : "=r"(x));
    return x;
}

static void init_inputs(void){
  for(int i=0;i<M;i++)
    for(int k=0;k<K;k++)
      A[i*K+k] = (elem_t)((i + 2*k) % 7 - 3);
  for(int k=0;k<K;k++)
    for(int j=0;j<N;j++)
      B[k*N+j] = (elem_t)(((k*3 + j) % 5) - 2);
  memset(C, 0, sizeof(C));
  memset(C_ref, 0, sizeof(C_ref));
}

static void cpu_ref(void){
  for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
      long long acc=0;
      for(int k=0;k<K;k++)
        acc += (long long)A[i*K+k] * (long long)B[k*N+j];
      C_ref[i*N+j] = (acc_t)acc;
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("Flush Gemmini TLB of stale virtual addresses\n");
    gemmini_flush(0);

    printf("Initialize input and output matrices in main memory\n");
    init_inputs();
    uint64_t c0_ref = rdcycle();
    cpu_ref();
    uint64_t c1_ref = rdcycle();
    printf("CPU matmul cycles: %lu\n", (unsigned long)(c1_ref - c0_ref));

    // Matrix strides (row-major order)
    const size_t A_stride = K;
    const size_t B_stride = N;
    const size_t C_stride = N;
    const size_t D_stride = N; 

    // ===== Baseline run: identity scaling + full_C=true (no quantization) =====
    uint64_t c0_gemm = rdcycle();
    tiled_matmul_auto(
        /*dim_I=*/M, /*dim_J=*/N, /*dim_K=*/K,
        /*A=*/(const elem_t*)A, /*B=*/(const elem_t*)B,
        /*D=*/NULL,              /*C=*/(void*)C,
        /*stride_A=*/A_stride, /*stride_B=*/B_stride, /*stride_D=*/D_stride, /*stride_C=*/C_stride,
        /*A_scale_factor=*/(scale_t)1.0f,
        /*B_scale_factor=*/(scale_t)1.0f,
        /*D_scale_factor=*/(scale_t)1.0f,
        /*act=*/NO_ACTIVATION,
        /*scale=*/(acc_scale_t)1.0f,     // no output quantization
        /*bert_scale=*/(acc_scale_t)1.0f,
        /*repeating_bias=*/false,
        /*transpose_A=*/false, /*transpose_B=*/false,
        /*full_C=*/true, /*low_D=*/false,
        /*weightA=*/0,
        /*tiled_matmul_type=*/WS
    );
    gemmini_fence();
    uint64_t c1_gemm = rdcycle();
    printf("Gemmini (baseline) cycles: %lu\n", (unsigned long)(c1_gemm - c0_gemm));

    // ===== Baseline correctness check =====
    long long diff=0, sum=0, sum_ref=0;
    for (int i = 0; i < M*N; i++) { if (C[i] != C_ref[i]) diff++; sum += C[i]; sum_ref += C_ref[i]; }
    if (diff==0) {
      printf("BASELINE PASS. checksum=%lld (ref=%lld)\n", sum, sum_ref);
    } else {
      printf("BASELINE FAIL. diff=%lld checksum=%lld ref=%lld\n", diff, sum, sum_ref);
    }

    // ===== Parameter sweep: A/B/D scaling × full_C × low_D =====
    // Explanation:
    // - Example sweep uses 3 scales for mvin: 1.0, 0.5, 0.25
    // - Output scale/bert_scale fixed to 1.0
    // - When full_C=false, accumulator results are quantized to elem_t before writeback
    // - D and low_D combinations test both low-precision and high-precision bias paths
    static elem_t D_low[M*N];
    static acc_t  D_high[M*N];
    for (int i = 0; i < M*N; i++) {
      int v = (i % 5) - 2;            // small bias in range -2..+2
      D_low[i]  = (elem_t)v;
      D_high[i] = (acc_t)v;
    }

    const scale_t mvin_scales[] = { (scale_t)1.0f, (scale_t)0.5f, (scale_t)0.25f };
    static const char* scale_names[] = { "1.0", "0.5", "0.25" };
    const int NUM_S = sizeof(mvin_scales)/sizeof(mvin_scales[0]);

    printf("\nCASE,A_scale,B_scale,D_scale,full_C,low_D,cycles\n");

    for (int ia = 0; ia < NUM_S; ia++) {
      for (int ib = 0; ib < NUM_S; ib++) {
        for (int id = 0; id < NUM_S; id++) {
          for (int fullC = 0; fullC <= 1; fullC++) {
            for (int lowD = 0; lowD <= 1; lowD++) {

              // Clear output buffer to reduce cache contamination effects
              memset(C, 0, sizeof(C));

              const void *D_ptr   = lowD ? (const void*)D_low  : (const void*)D_high;

              uint64_t t0 = rdcycle();
              tiled_matmul_auto(
                  /*dim_I=*/M, /*dim_J=*/N, /*dim_K=*/K,
                  /*A=*/(const elem_t*)A, /*B=*/(const elem_t*)B,
                  /*D=*/D_ptr,            /*C=*/(void*)C,
                  /*stride_A=*/A_stride, /*stride_B=*/B_stride, /*stride_D=*/D_stride, /*stride_C=*/C_stride,
                  /*A_scale_factor=*/mvin_scales[ia],
                  /*B_scale_factor=*/mvin_scales[ib],
                  /*D_scale_factor=*/mvin_scales[id],
                  /*act=*/NO_ACTIVATION,
                  /*scale=*/(acc_scale_t)1.0f,         // output scale = 1.0 (no requantization)
                  /*bert_scale=*/(acc_scale_t)1.0f,
                  /*repeating_bias=*/false,
                  /*transpose_A=*/false, /*transpose_B=*/false,
                  /*full_C=*/fullC, /*low_D=*/lowD,
                  /*weightA=*/0,
                  /*tiled_matmul_type=*/WS
              );
              gemmini_fence();
              uint64_t t1 = rdcycle();

              printf("sweep,%s,%s,%s,%d,%d,%llu\n",
                scale_names[ia], scale_names[ib], scale_names[id],
                fullC, lowD,
                (unsigned long long)(t1 - t0));
            }
          }
        }
      }
    }
}