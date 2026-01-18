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
#include "include/gemmini_testutils_fp.h"
//#include "64_by_64_matrices_1.h"

#define dual_array_dim 4

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

static void fp_counter_configure_extra(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0xff) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  fp_gemmini_counter_access(placeholder, config_reg);
}

static uint32_t fp_counter_read_extra(size_t index) {
  uint32_t config_reg = (index & 0xff) << 4;
  uint32_t res;
  fp_gemmini_counter_access(res, config_reg);
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

  char* counter_names[60] = {
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
	"PROFILING_SUM_A",
	"PROFILING_SUM_B",
	"PROFILING_SUM_D",
};

int counter_val_start[57];
int counter_val_end[57];

uint32_t profiling_start_a;
uint32_t profiling_end_a;

uint32_t profiling_start_b;
uint32_t profiling_end_b;

uint32_t profiling_start_d;
uint32_t profiling_end_d;

  for (int i = 0; i < 60; ++i) {

  	counter_configure_extra(i, i + 1);

  }

	printf("dual_array_dim: %d\n\n", dual_array_dim);
	gemmini_flush(0);
	fp_gemmini_flush(0);

	counter_reset();
  	fp_counter_reset();


	elem_t A[dual_array_dim][dual_array_dim];
        elem_t B[dual_array_dim][dual_array_dim];
        elem_t D[dual_array_dim][dual_array_dim];

        acc_t C[dual_array_dim][dual_array_dim];

        for (size_t i = 0; i < dual_array_dim; ++i) {
          for (size_t j = 0; j < dual_array_dim; ++j) {
            A[i][j] = (rand() % 4) * 16;
            B[i][j] = (rand() % 4) * 16;
            D[i][j] = (rand() % 4) * 16;
          }
        }

	fp_elem_t FP_A[dual_array_dim][dual_array_dim];
        fp_elem_t FP_B[dual_array_dim][dual_array_dim];
        fp_elem_t FP_D[dual_array_dim][dual_array_dim];

        fp_acc_t FP_C[dual_array_dim][dual_array_dim];

        for (size_t i = 0; i < dual_array_dim; ++i) {
          for (size_t j = 0; j < dual_array_dim; ++j) {
            FP_A[i][j] = fp_rand_double();
            FP_B[i][j] = fp_rand_double();
            FP_D[i][j] = fp_rand_double();
          }
        }

//       uint32_t A_addr = 0;
//       uint32_t B_addr = DIM;
//       uint32_t D_addr = 2*DIM;
//       uint32_t C_addr = 3*DIM;

      // Calculate the proper destination addresses of everything

    printf("A:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			A[i][j] = A[i][j] * 2;
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("B:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			B[i][j] = B[i][j] * 2;
			printf("%d ", B[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("D:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			D[i][j] = D[i][j] * 2;
			printf("%d ", D[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("FP_A:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			A[i][j] = A[i][j] * 2;
			printf("%x ", fp_elem_t_to_elem_t_bits(FP_A[i][j]));
		}
		printf("\n");
	}
	printf("\n");

	printf("FP_B:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			B[i][j] = B[i][j] * 2;
			printf("%x ", fp_elem_t_to_elem_t_bits(FP_B[i][j]));
		}
		printf("\n");
	}
	printf("\n");

	printf("FP_D:\n");
	for (size_t i = 0; i < dual_array_dim; ++i) {
		for (size_t j = 0; j < dual_array_dim; ++j) {
//			D[i][j] = D[i][j] * 2;
			printf("%x ", fp_elem_t_to_elem_t_bits(FP_D[i][j]));
		}
		printf("\n");
	}
	printf("\n");

	
	for (int i = 0; i < 57; ++i) { //assuming the compiler will unroll the loop
		counter_val_start[i] = counter_read_extra(i);
    	}
	unsigned long start = read_cycles();
	profiling_start_a = counter_read_extra(57);
	profiling_start_b = counter_read_extra(58);
	profiling_start_d = counter_read_extra(59);
	// TODO:
	// MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
	// MAKE A "COUNTER" VERSION OF THIS FUNCTION
	// FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
	//printf("Here A");

	printf("dual_array_dim: %d\n\n", dual_array_dim);

	tiled_matmul_auto(dual_array_dim, dual_array_dim, dual_array_dim,
            (elem_t*)A, (elem_t*)B,
	    &D[0][0], (elem_t*)C,
            dual_array_dim, dual_array_dim, dual_array_dim, dual_array_dim,
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
    int total_macs = dual_array_dim * dual_array_dim * dual_array_dim;
    int ideal_cycles = total_macs / (DIM * DIM);
    int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
        //

    for (int i = 0; i < 57; ++i) { //assuming the compiler will unroll the loop
	counter_val_end[i] = counter_read_extra(i);
    }
	profiling_end_a = counter_read_extra(57);
	profiling_end_b = counter_read_extra(58);
	profiling_end_d = counter_read_extra(59);
	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i < 57; ++i) {
		printf("DIM %d: %s, %d\n", dual_array_dim, counter_names[i], counter_val_end[i] - counter_val_start[i]);
	}
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[57], profiling_end_a, profiling_start_a);
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[58], profiling_end_b, profiling_start_b);
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[59], profiling_end_d, profiling_start_d);
	printf("\n\n\n");

	//counter_reset();

      printf("Moved out\n");
          printf("C:\n");
	  for (size_t i = 0; i < dual_array_dim; ++i) {
    	    for (size_t j = 0; j < dual_array_dim; ++j) {
      	      printf("%d ", C[i][j]);
	    }
    	    printf("\n");
  	  }
          printf("\n");


	  //FLOATING POINT

      for (int i = 0; i < 57; ++i) { //assuming the compiler will unroll the loop
		counter_val_start[i] = fp_counter_read_extra(i);
    	}
	start = read_cycles();
	profiling_start_a = fp_counter_read_extra(57);
	profiling_start_b = fp_counter_read_extra(58);
	profiling_start_d = fp_counter_read_extra(59);
	// TODO:
	// MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
	// MAKE A "COUNTER" VERSION OF THIS FUNCTION
	// FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
	//printf("Here A");
	fp_tiled_matmul_auto(dual_array_dim, dual_array_dim, dual_array_dim,
            (fp_elem_t*)FP_A, (fp_elem_t*)FP_B,
	    &FP_D[0][0], (fp_elem_t*)FP_C,
            dual_array_dim, dual_array_dim, dual_array_dim, dual_array_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, FP_ACC_SCALE_IDENTITY, 0, 
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
    total_macs = dual_array_dim * dual_array_dim * dual_array_dim;
    ideal_cycles = total_macs / (FP_DIM * FP_DIM);
    utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
        //

    for (int i = 0; i < 57; ++i) { //assuming the compiler will unroll the loop
	counter_val_end[i] = fp_counter_read_extra(i);
    }
	profiling_end_a = fp_counter_read_extra(57);
	profiling_end_b = fp_counter_read_extra(58);
	profiling_end_d = fp_counter_read_extra(59);
	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i < 57; ++i) {
		printf("DIM %d: %s, %d\n", dual_array_dim, counter_names[i], counter_val_end[i] - counter_val_start[i]);
	}
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[57], profiling_end_a, profiling_start_a);
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[58], profiling_end_b, profiling_start_b);
		printf("DIM %d: %s, end: %x start: %x\n", dual_array_dim, counter_names[59], profiling_end_d, profiling_start_d);
	printf("\n\n\n");

	//counter_reset();

      printf("Moved out\n");
          printf("FP_C:\n");
	  for (size_t i = 0; i < dual_array_dim; ++i) {
    	    for (size_t j = 0; j < dual_array_dim; ++j) {
      	      printf("%x ", fp_acc_t_to_acc_t_bits(FP_C[i][j]));
	    }
    	    printf("\n");
  	  }
          printf("\n");

  }