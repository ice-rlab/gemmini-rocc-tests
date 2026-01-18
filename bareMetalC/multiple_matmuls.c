#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini_testutils.h"
//#include "64_by_64_matrices_1.h"

// #ifndef M_DIM
// #define M_DIM 16
// #endif

// normal counter funcitons only works if you stick to the default number of counters (8)
static void counter_configure_extra(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0xff) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

static uint32_t counter_read_extra(size_t index) {
  uint32_t config_reg = (index & 0xff) << 4;
  uint32_t res;
  gemmini_counter_access(res, config_reg);
  return res;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  //static elem_t ZERO[DIM][DIM];

  char* counter_names[58] = {
    "MAIN_LD_CYCLES",
    "MAIN_ST_CYCLES",
    "MAIN_EX_CYCLES",
    "MAIN_LD_ST_CYCLES",
    "MAIN_LD_EX_CYCLES",
    "MAIN_ST_EX_CYCLES",
    "MAIN_LD_ST_EX_CYCLES",
    "LOAD_DMA_WAIT_CYCLE",
    "LOAD_ACTIVE_CYCLE",
    "LOAD_SCRATCHPAD_WAIT_CYCLE",
    "STORE_DMA_WAIT_CYCLE",
    "STORE_ACTIVE_CYCLE",
    "STORE_POOLING_CYCLE",
    "STORE_SCRATCHPAD_WAIT_CYCLE",
    "DMA_TLB_MISS_CYCLE",
    "DMA_TLB_HIT_REQ",
    "DMA_TLB_TOTAL_REQ",
    "RDMA_ACTIVE_CYCLE",
    "RDMA_TLB_WAIT_CYCLES",
    "RDMA_TL_WAIT_CYCLES",
    "WDMA_ACTIVE_CYCLE",
    "WDMA_TLB_WAIT_CYCLES",
    "WDMA_TL_WAIT_CYCLES",
    "EXE_ACTIVE_CYCLE",
    "EXE_FLUSH_CYCLE",
    "EXE_CONTROL_Q_BLOCK_CYCLE",
    "EXE_PRELOAD_HAZ_CYCLE",
    "EXE_OVERLAP_HAZ_CYCLE",
    "SCRATCHPAD_A_WAIT_CYCLE",
    "SCRATCHPAD_B_WAIT_CYCLE",
    "SCRATCHPAD_D_WAIT_CYCLE",
    "ACC_A_WAIT_CYCLE",
    "ACC_B_WAIT_CYCLE",
    "ACC_D_WAIT_CYCLE",
    "A_GARBAGE_CYCLES",
    "B_GARBAGE_CYCLES",
    "D_GARBAGE_CYCLES",
    "IM2COL_MEM_CYCLES",
    "IM2COL_ACTIVE_CYCLES",
    "IM2COL_TRANSPOSER_WAIT_CYCLE",
    "RESERVATION_STATION_FULL_CYCLES",
    "RESERVATION_STATION_ACTIVE_CYCLES",
    "LOOP_MATMUL_ACTIVE_CYCLES",
    "TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES",
    "EXE_ONLY_PRELOAD_CYCLE",
    "EXE_OVERLAPING_CYCLE",
    "EXE_ONLY_MATMUL_CYCLE",
    "MATMUL_IN_PROGRESS",
    "MAC_BUSY",
    "RESERVATION_STATION_LD_COUNT",
    "RESERVATION_STATION_ST_COUNT",
    "RESERVATION_STATION_EX_COUNT",
    "RDMA_BYTES_REC",
    "WDMA_BYTES_SENT",
    "RDMA_TOTAL_LATENCY",
    "WDMA_TOTAL_LATENCY",
    "TOTAL_MACS",
    "TOTAL_MACS2"
};

int counter_val_start[58];
int counter_val_end[58];

  for (int i = 0; i < 58; ++i) {
        counter_configure_extra(i, i + 1);
  }
        printf("19 by 19\n\n");
        gemmini_flush(0);

        counter_reset();

        gemmini_config_ld(19 * sizeof(elem_t));

        elem_t A1[19][19];
        elem_t B1[19][19];
        elem_t D1[19][19];

        acc_t C1[19][19];

	elem_t A2[40][70];
        elem_t B2[70][35];
        elem_t D2[40][35];

        acc_t C2[40][35];

  	elem_t A3[10][1];
        elem_t B3[1][5];
        elem_t D3[10][5];

        acc_t C3[10][5];

        for (size_t i = 0; i < 19; ++i) {
          for (size_t j = 0; j < 19; ++j) {
            A1[i][j] = ((rand() % 512) - 256) & 0xff;
            B1[i][j] = ((rand() % 512) - 256) & 0xff;
            D1[i][j] = ((rand() % 512) - 256) & 0xff;
          }
        }

      // Calculate the proper destination addresses of everything

    printf("A1:\n");
        for (size_t i = 0; i < 19; ++i) {
                for (size_t j = 0; j < 19; ++j) {
//                      A[i][j] = A[i][j] * 2;
                        printf("%d ", A1[i][j]);
                }
                printf("\n");
        }
        printf("\n");

        printf("B1:\n");
        for (size_t i = 0; i < 19; ++i) {
                for (size_t j = 0; j < 19; ++j) {
//                      B[i][j] = B[i][j] * 2;
                        printf("%d ", B1[i][j]);
                }
                printf("\n");
        }
        printf("\n");

        printf("D1:\n");
        for (size_t i = 0; i < 19; ++i) {
                for (size_t j = 0; j < 19; ++j) {
//                      D[i][j] = D[i][j] * 2;
                        printf("%d ", D1[i][j]);
                }
                printf("\n");
        }
        printf("\n");

        for (int i = 0; i < 58; ++i) { //assuming the compiler will unroll the loop
                counter_val_start[i] = counter_read_extra(i);
        }
        unsigned long start = read_cycles();
        // TODO:
        // MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
        // MAKE A "COUNTER" VERSION OF THIS FUNCTION
        // FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
        //printf("Here A");
        tiled_matmul_auto(19, 19, 19,
            (elem_t*)A1, (elem_t*)B1,
            &D1[0][0], (elem_t*)C1,
            19, 19, 19, 19,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0,
            false,
            0, 0,
            true, true, //reacing out C in acc_t form, D is in input type form
            0,
            WS); //USED DEFAULT VALUES FROM MATMUL_TEMPLATE
        //printf("Here B");
        /*
        void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA, (seems like weightA is never used at all though???)
        enum tiled_matmul_type_t tiled_matmul_type) {
        */

      //gemmini_fence();
      unsigned long end = read_cycles();
        printf("STATS\n\nCycles taken: %u\n", end-start);
    int total_macs = 19 * 19 * 19;
    int ideal_cycles = total_macs / (16 * 16);
    int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
       
        //printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

        
        //counter_reset();

      printf("Moved out\n");
          printf("C1:\n");
          for (size_t i = 0; i < 19; ++i) {
            for (size_t j = 0; j < 19; ++j) {
              printf("%d ", C1[i][j]);
            }
            printf("\n");
          }
          printf("\n");



	    for (size_t i = 0; i < 40; ++i) {
            for (size_t j = 0; j < 35; ++j) {
				for (size_t k = 0; k < 70; ++k) {
					A2[i][k] = ((rand() % 512) - 256) & 0xff;
					B2[k][j] = ((rand() % 512) - 256) & 0xff;
				}
				D2[i][j] = ((rand() % 512) - 256) & 0xff;
            }
        }

      // Calculate the proper destination addresses of everything

    printf("A2:\n");
        for (size_t i = 0; i < 40; ++i) {
                for (size_t k = 0; k < 70; ++k) {
//                      A[i][j] = A[i][j] * 2;
                        printf("%d ", A2[i][k]);
                }
                printf("\n");
        }
        printf("\n");

        printf("B2:\n");
        for (size_t k = 0; k < 70; ++k) {
                for (size_t j = 0; j < 35; ++j) {
//                      B[i][j] = B[i][j] * 2;
                        printf("%d ", B2[k][j]);
                }
                printf("\n");
        }
        printf("\n");

        printf("D2:\n");
        for (size_t i = 0; i < 40; ++i) {
                for (size_t j = 0; j < 35; ++j) {
//                      D[i][j] = D[i][j] * 2;
                        printf("%d ", D2[i][j]);
                }
                printf("\n");
        }
        printf("\n");

        start = read_cycles();
        // TODO:
        // MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
        // MAKE A "COUNTER" VERSION OF THIS FUNCTION
        // FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
        //printf("Here A");
        tiled_matmul_auto(40, 35, 70,
            (elem_t*)A2, (elem_t*)B2,
            &D2[0][0], (elem_t*)C2,
            70, 35, 35, 35,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0,
            false,
            0, 0,
            true, true, //reacing out C in acc_t form, D is in input type form
            0,
            WS); //USED DEFAULT VALUES FROM MATMUL_TEMPLATE
        //printf("Here B");
        /*
        void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA, (seems like weightA is never used at all though???)
        enum tiled_matmul_type_t tiled_matmul_type) {
        */

      //gemmini_fence();
      end = read_cycles();
        printf("STATS\n\nCycles taken: %u\n", end-start);
    total_macs = 40 * 70 * 35;
    ideal_cycles = total_macs / (16 * 16);
    utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);

       
        //printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

        
        //counter_reset();

      printf("Moved out\n");
          printf("C2:\n");
          for (size_t i = 0; i < 40; ++i) {
            for (size_t j = 0; j < 35; ++j) {
              printf("%d ", C2[i][j]);
            }
            printf("\n");
          }
          printf("\n");



	    for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < 5; ++j) {
				for (size_t k = 0; k < 1; ++k) {
					A3[i][k] = ((rand() % 512) - 256) & 0xff;
					B3[k][j] = ((rand() % 512) - 256) & 0xff;
				}
				D3[i][j] = ((rand() % 512) - 256) & 0xff;
            }
        }

      // Calculate the proper destination addresses of everything

    printf("A3:\n");
        for (size_t i = 0; i < 10; ++i) {
                for (size_t k = 0; k < 1; ++k) {
//                      A[i][j] = A[i][j] * 2;
                        printf("%d ", A3[i][k]);
                }
                printf("\n");
        }
        printf("\n");

        printf("B3:\n");
        for (size_t k = 0; k < 1; ++k) {
                for (size_t j = 0; j < 5; ++j) {
//                      B[i][j] = B[i][j] * 2;
                        printf("%d ", B3[k][j]);
                }
                printf("\n");
        }
        printf("\n");

        printf("D3:\n");
        for (size_t i = 0; i < 10; ++i) {
                for (size_t j = 0; j < 5; ++j) {
//                      D[i][j] = D[i][j] * 2;
                        printf("%d ", D3[i][j]);
                }
                printf("\n");
        }
        printf("\n");

        start = read_cycles();
        // TODO:
        // MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
        // MAKE A "COUNTER" VERSION OF THIS FUNCTION
        // FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
        //printf("Here A");
        tiled_matmul_auto(10, 5, 1,
            (elem_t*)A3, (elem_t*)B3,
            &D3[0][0], (elem_t*)C3,
            1, 5, 5, 5,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0,
            false,
            0, 0,
            true, true, //reacing out C in acc_t form, D is in input type form
            0,
            WS); //USED DEFAULT VALUES FROM MATMUL_TEMPLATE
        //printf("Here B");
        /*
        void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA, (seems like weightA is never used at all though???)
        enum tiled_matmul_type_t tiled_matmul_type) {
        */

      //gemmini_fence();
      end = read_cycles();
        printf("STATS\n\nCycles taken: %u\n", end-start);
    total_macs = 10 * 1 * 5;
    ideal_cycles = total_macs / (16 * 16);
    utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
       
        //printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);
		for (int i = 0; i < 58; ++i) { //assuming the compiler will unroll the loop
                counter_val_end[i] = counter_read_extra(i);
        }
        
        //counter_reset();

      printf("Moved out\n");
          printf("C3:\n");
          for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < 5; ++j) {
              printf("%d ", C3[i][j]);
            }
            printf("\n");
          }
          printf("\n");



	  for (int i = 0; i < 58; ++i) {
                printf("%s, %d\n", counter_names[i], counter_val_end[i] - counter_val_start[i]);
        }

        printf("\n\n\n");

  }





