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

  char* counter_names[66] = {
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
	"STREAM_READER_BUSY",
	"STREAM_WRITER_BUSY",
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
    "BYTES_LOADED_A",
    "BYTES_LOADED_B",
    "BYTES_LOADED_D",
    "BYTES_READ"
};

int counter_val_start[66];
int counter_val_end[66];

  for (int i = 0; i < 66; ++i) {
        counter_configure_extra(i, i + 1);
  }
        printf("MVIN MVOUT LARGE MATRIX");
        gemmini_flush(0);

        counter_reset();

  	//

        //static elem_t A[DIM][DIM*1000] row_align(1); //intentionally made it a bit less than max capacity
	//static acc_t B[DIM][DIM*500] row_align(1);
	//static elem_t C[DIM][DIM*1000] row_align(1);

//         for (size_t i = 0; i < DIM; ++i) {
//           for (size_t j = 0; j < (DIM*1000); ++j) {
//             A[i][j] = ((rand() % 512) - 256) & 0xff;
// //	    B[i][j] = ((rand() % 512) - 256) & 0xff;
//           }
//         }

  	//default accumulator has 65536 bytes

	//with default accType that is 16384 matrix elements

	//a 64, 16 by 16 matricies of acc type would then fill the whole accumulator

	static acc_t A[DIM][DIM*56] row_align(1); //intentionally made it a bit less than max capacity
	static acc_t B[DIM][DIM*56] row_align(1);
	static acc_t C[DIM][DIM*56] row_align(1);

        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < (DIM*56); ++j) {
            A[i][j] = ((rand() % 512) - 256) & 0xff;
	    B[i][j] = ((rand() % 512) - 256) & 0xff;
          }
        }



	//gemmini_extended3_config_ld(DIM * 1000 * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);

	gemmini_extended5_config_ld(DIM * 56 * sizeof(acc_t), MVIN_SCALE_IDENTITY, 0, DIM, 1, 0);
	gemmini_extended5_config_ld(DIM * 56 * sizeof(acc_t), MVIN_SCALE_IDENTITY, 0, DIM, 1, 1);

	
	gemmini_config_st(DIM * 56 * sizeof(acc_t));

        //uint32_t A_addr = 0;
	uint32_t A_addr = 0x80000000;
	uint32_t B_addr = 0xC0000000; //Also accumulate on top uint32_t B_addr = 0xC0000000;
	uint32_t C_addr = 0xA0000000;
//       uint32_t B_addr = DIM;
//       uint32_t D_addr = 2*DIM;
//       uint32_t C_addr = 3*DIM;

      // Calculate the proper destination addresses of everything

        for (int i = 0; i < 66; ++i) { //assuming the compiler will unroll the loop
                counter_val_start[i] = counter_read_extra(i);
        }
        unsigned long start = read_cycles();

	// gemmini_extended_mvin(A, A_addr, DIM * 1000, DIM);
	// gemmini_fence();
	// gemmini_extended_mvout(C, A_addr, DIM * 1000, DIM);
	// gemmini_fence();
	gemmini_extended_mvin(A, A_addr, DIM * 56, DIM);
	gemmini_extended_mvin2(B, B_addr, DIM * 56, DIM);
	gemmini_fence();
	gemmini_preload_zeros(C);
	gemmini_compute_preloaded(A, B);
	gemmini_fence();
	gemmini_extended_mvout(C, C_addr, DIM * 56, DIM);
	gemmini_fence();

      unsigned long end = read_cycles();
        printf("STATS\n\nCycles taken: %u\n", end-start);
//     const int total_macs = M_DIM * M_DIM * M_DIM;
//     const int ideal_cycles = total_macs / (DIM * DIM);
//     const int utilization = 100 * ideal_cycles / (end-start);
//     printf("Utilization: %d%%\n\n\n", utilization);
	
	

    for (int i = 0; i < 66; ++i) { //assuming the compiler will unroll the loop
        counter_val_end[i] = counter_read_extra(i);
    }
       
        //printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

        for (int i = 0; i < 66; ++i) {
                printf("DIM %d: %s, %d\n", DIM, counter_names[i], counter_val_end[i] - counter_val_start[i]);
        }
	printf("RESULT:\n\n");
	for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < (DIM*56); ++j) {
            printf("%d ", C[i][j]); //C[i][j] = ((rand() % 512) - 256) & 0xff;
//	    B[i][j] = ((rand() % 512) - 256) & 0xff;
          }
	  printf("\n");
        }
	printf("\n");

  }





