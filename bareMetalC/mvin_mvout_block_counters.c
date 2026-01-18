// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

//Scratchpad is 2^18 bytes big
//each element is 1 byte big
//2^10 matrcies fit in the scratchpad

//#define N 1000

#define BLOCK_STRIDE (DIM*2)
#define COLS (DIM*MAX_BLOCK_LEN)

// #if (N*DIM) > (BANK_NUM*BANK_ROWS)
// #error not enough scratchpad space
// #endif

void printMatrixFull(elem_t m[DIM][COLS]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < COLS; ++j)
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

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

  // printf("Flush\n");
  gemmini_flush(0);

  counter_reset();

  gemmini_extended4_config_ld(COLS * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, BLOCK_STRIDE, 0);
  gemmini_config_st(COLS * sizeof(elem_t));

  static elem_t In[250][DIM][COLS] row_align(1);
  static elem_t Out[250][DIM][COLS] row_align(1);
for (size_t k = 0; k < 250; ++k)
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < COLS; ++j)
      In[k][i][j] = i*COLS + j;

for (int i = 0; i < 66; ++i) { //assuming the compiler will unroll the loop
                counter_val_start[i] = counter_read_extra(i);
        }
        unsigned long start_mvin = read_cycles();
for (size_t n = 0; n < 250; ++n) {
  gemmini_block_mvin(In[n], n*DIM*MAX_BLOCK_LEN, MAX_BLOCK_LEN);
}
  gemmini_fence();
unsigned long end_mvin = read_cycles();
unsigned long start_mvout = read_cycles();
  for (size_t i = 0; i < 250; ++i) {
	for (size_t n = 0; n < MAX_BLOCK_LEN; ++n) {
	  gemmini_mvout((elem_t*)Out + ((MAX_BLOCK_LEN * i) + n) * DIM, ((MAX_BLOCK_LEN * i) + n)*BLOCK_STRIDE);
	}
  }

  gemmini_fence();

unsigned long end_mvout = read_cycles();
 printf("\n\nMVIN ycles taken: %u\n", end_mvin-start_mvin);
 printf("\n\nMVOUT ycles taken: %u\n", end_mvout-start_mvout);
for (int i = 0; i < 66; ++i) { //assuming the compiler will unroll the loop
        counter_val_end[i] = counter_read_extra(i);
    }

for (int i = 0; i < 66; ++i) {
                printf("%s, %d\n", counter_names[i], counter_val_end[i] - counter_val_start[i]);
        }

//    if (!MAT_IS_EQUAL(DIM, COLS, In, Out)) {
//     printf("Matrix:\n");
//     printMatrixFull(In);
//     printf("\nMatrix output:\n");
//     printMatrixFull(Out);
//     printf("\n");

//     exit(1);
//   }

  printf("PASS");

  exit(0);
}

