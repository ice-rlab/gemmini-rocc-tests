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

#ifndef ARRAY_DIM
#define ARRAY_DIM DIM
#endif

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

  char* counter_names[51] = {
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
    "RESERVATION_STATION_LD_COUNT",
    "RESERVATION_STATION_ST_COUNT",
    "RESERVATION_STATION_EX_COUNT",
    "RDMA_BYTES_REC",
    "WDMA_BYTES_SENT",
    "RDMA_TOTAL_LATENCY",
    "WDMA_TOTAL_LATENCY"
};

int counter_val_start[51];
int counter_val_end[51];

  

  counter_configure_extra(0, MAIN_LD_CYCLES);
  counter_configure_extra(1, MAIN_ST_CYCLES);
  counter_configure_extra(2, MAIN_EX_CYCLES);
  counter_configure_extra(3, MAIN_LD_ST_CYCLES);
  counter_configure_extra(4, MAIN_LD_EX_CYCLES);
  counter_configure_extra(5, MAIN_ST_EX_CYCLES);
  counter_configure_extra(6, MAIN_LD_ST_EX_CYCLES);

  counter_configure_extra(7, LOAD_DMA_WAIT_CYCLE);
  counter_configure_extra(8, LOAD_ACTIVE_CYCLE);
  counter_configure_extra(9, LOAD_SCRATCHPAD_WAIT_CYCLE);

  counter_configure_extra(10, STORE_DMA_WAIT_CYCLE);
  counter_configure_extra(11, STORE_ACTIVE_CYCLE);
  counter_configure_extra(12, STORE_POOLING_CYCLE);
  counter_configure_extra(13, STORE_SCRATCHPAD_WAIT_CYCLE);

  counter_configure_extra(14, DMA_TLB_MISS_CYCLE);
  counter_configure_extra(15, DMA_TLB_HIT_REQ);
  counter_configure_extra(16, DMA_TLB_TOTAL_REQ);

  counter_configure_extra(17, RDMA_ACTIVE_CYCLE);
  counter_configure_extra(18, RDMA_TLB_WAIT_CYCLES);
  counter_configure_extra(19, RDMA_TL_WAIT_CYCLES);

  counter_configure_extra(20, WDMA_ACTIVE_CYCLE);
  counter_configure_extra(21, WDMA_TLB_WAIT_CYCLES);
  counter_configure_extra(22, WDMA_TL_WAIT_CYCLES);

  counter_configure_extra(23, EXE_ACTIVE_CYCLE);
  counter_configure_extra(24, EXE_FLUSH_CYCLE);
  counter_configure_extra(25, EXE_CONTROL_Q_BLOCK_CYCLE);
  counter_configure_extra(26, EXE_PRELOAD_HAZ_CYCLE);
  counter_configure_extra(27, EXE_OVERLAP_HAZ_CYCLE);

  counter_configure_extra(28, SCRATCHPAD_A_WAIT_CYCLE);
  counter_configure_extra(29, SCRATCHPAD_B_WAIT_CYCLE);
  counter_configure_extra(30, SCRATCHPAD_D_WAIT_CYCLE);

  counter_configure_extra(31, ACC_A_WAIT_CYCLE);
  counter_configure_extra(32, ACC_B_WAIT_CYCLE);
  counter_configure_extra(33, ACC_D_WAIT_CYCLE);

  counter_configure_extra(34, A_GARBAGE_CYCLES);
  counter_configure_extra(35, B_GARBAGE_CYCLES);
  counter_configure_extra(36, D_GARBAGE_CYCLES);

  counter_configure_extra(37, IM2COL_MEM_CYCLES);
  counter_configure_extra(38, IM2COL_ACTIVE_CYCLES);
  counter_configure_extra(39, IM2COL_TRANSPOSER_WAIT_CYCLE);

  counter_configure_extra(40, RESERVATION_STATION_FULL_CYCLES);
  counter_configure_extra(41, RESERVATION_STATION_ACTIVE_CYCLES);

  counter_configure_extra(42, LOOP_MATMUL_ACTIVE_CYCLES);
  counter_configure_extra(43, TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES);

  counter_configure_extra(44, RESERVATION_STATION_LD_COUNT);
  counter_configure_extra(45, RESERVATION_STATION_ST_COUNT);
  counter_configure_extra(46, RESERVATION_STATION_EX_COUNT);

  counter_configure_extra(47, RDMA_BYTES_REC);
  counter_configure_extra(48, WDMA_BYTES_SENT);

  counter_configure_extra(49, RDMA_TOTAL_LATENCY);
  counter_configure_extra(50, WDMA_TOTAL_LATENCY);


	printf("ARRAY_DIM: %d\n\n", ARRAY_DIM);
	gemmini_flush(0);

	counter_reset();

  	gemmini_config_ld(ARRAY_DIM * sizeof(elem_t));

	elem_t A[ARRAY_DIM][ARRAY_DIM];
        elem_t B[ARRAY_DIM][ARRAY_DIM];
        elem_t D[ARRAY_DIM][ARRAY_DIM];

        elem_t C[ARRAY_DIM][ARRAY_DIM];

        for (size_t i = 0; i < ARRAY_DIM; ++i) {
          for (size_t j = 0; j < ARRAY_DIM; ++j) {
            A[i][j] = (rand() % 4);
            B[i][j] = (rand() % 4);
            D[i][j] = (rand() % 4);
          }
        }

//       uint32_t A_addr = 0;
//       uint32_t B_addr = DIM;
//       uint32_t D_addr = 2*DIM;
//       uint32_t C_addr = 3*DIM;

      // Calculate the proper destination addresses of everything

    printf("A:\n");
	for (size_t i = 0; i < ARRAY_DIM; ++i) {
		for (size_t j = 0; j < ARRAY_DIM; ++j) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("B:\n");
	for (size_t i = 0; i < ARRAY_DIM; ++i) {
		for (size_t j = 0; j < ARRAY_DIM; ++j) {
			printf("%d ", B[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("D:\n");
	for (size_t i = 0; i < ARRAY_DIM; ++i) {
		for (size_t j = 0; j < ARRAY_DIM; ++j) {
			printf("%d ", D[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	
	for (int i = 0; i < 51; ++i) { //assuming the compiler will unroll the loop
		counter_val_start[i] = counter_read_extra(i);
    	}
	unsigned long start = read_cycles();
	// TODO:
	// MAKE SURE TILED MATMUL AUTO PARAMETERS ARE CORRECT
	// MAKE A "COUNTER" VERSION OF THIS FUNCTION
	// FIX BUG WHERE THE OUTPUT IS JUST MAXED OUT
	//printf("Here A");
	tiled_matmul_auto(ARRAY_DIM, ARRAY_DIM, ARRAY_DIM,
            (elem_t*)A, (elem_t*)B,
	    &D[0][0], (elem_t*)C,
            ARRAY_DIM, ARRAY_DIM, ARRAY_DIM, ARRAY_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0, 
	    false,
            0, 0,
            false, true,
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
    const int total_macs = ARRAY_DIM * ARRAY_DIM * ARRAY_DIM;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
        //

    for (int i = 0; i < 51; ++i) { //assuming the compiler will unroll the loop
	counter_val_end[i] = counter_read_extra(i);
    }
	
	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i <= 50; ++i) {
		printf("DIM %d: %s, %d\n", ARRAY_DIM, counter_names[i], counter_val_end[i] - counter_val_start[i]);
	}

	printf("\n\n\n");

	//counter_reset();

      printf("Moved out\n");
          printf("C:\n");
	  for (size_t i = 0; i < ARRAY_DIM; ++i) {
    	    for (size_t j = 0; j < ARRAY_DIM; ++j) {
      	      printf("%d ", C[i][j]);
	    }
    	    printf("\n");
  	  }
          printf("\n");
  }