// See LICENSE for license details.

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


#ifdef FAST
#define AINIT RELU
#define SINIT 12
#define N 1
#else
#define AINIT NO_ACTIVATION
#define SINIT 0
#define N 2
#endif


void operands(int c, int * a, int * b, int * d) {
  *d = c % N;
  *b = (c / N) % N;
  *a = c / (N*N);
}

#if 3*N*DIM > (BANK_NUM * BANK_ROWS) || N*N*N*DIM > ACC_ROWS
//#error scratchpad or accumulator not big enough
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  static elem_t ZERO[DIM][DIM];

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

int counter_val[51];

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));

//   counter_configure(0, EXE_ACTIVE_CYCLE);
//   counter_configure(1, EXE_FLUSH_CYCLE);
//   counter_configure(2, EXE_CONTROL_Q_BLOCK_CYCLE);
//   counter_configure(3, EXE_PRELOAD_HAZ_CYCLE);
//   counter_configure(4, EXE_OVERLAP_HAZ_CYCLE);
//   counter_configure(5, MAIN_EX_CYCLES);

  counter_configure(0, MAIN_LD_CYCLES);
  counter_configure(1, MAIN_ST_CYCLES);
  counter_configure(2, MAIN_EX_CYCLES);
  counter_configure(3, MAIN_LD_ST_CYCLES);
  counter_configure(4, MAIN_LD_EX_CYCLES);
  counter_configure(5, MAIN_ST_EX_CYCLES);
  counter_configure(6, MAIN_LD_ST_EX_CYCLES);

  counter_configure(7, LOAD_DMA_WAIT_CYCLE);
  counter_configure(8, LOAD_ACTIVE_CYCLE);
  counter_configure(9, LOAD_SCRATCHPAD_WAIT_CYCLE);

  counter_configure(10, STORE_DMA_WAIT_CYCLE);
  counter_configure(11, STORE_ACTIVE_CYCLE);
  counter_configure(12, STORE_POOLING_CYCLE);
  counter_configure(13, STORE_SCRATCHPAD_WAIT_CYCLE);

  counter_configure(14, DMA_TLB_MISS_CYCLE);
  counter_configure(15, DMA_TLB_HIT_REQ);
  counter_configure(16, DMA_TLB_TOTAL_REQ);

  counter_configure(17, RDMA_ACTIVE_CYCLE);
  counter_configure(18, RDMA_TLB_WAIT_CYCLES);
  counter_configure(19, RDMA_TL_WAIT_CYCLES);

  counter_configure(20, WDMA_ACTIVE_CYCLE);
  counter_configure(21, WDMA_TLB_WAIT_CYCLES);
  counter_configure(22, WDMA_TL_WAIT_CYCLES);

  counter_configure(23, EXE_ACTIVE_CYCLE);
  counter_configure(24, EXE_FLUSH_CYCLE);
  counter_configure(25, EXE_CONTROL_Q_BLOCK_CYCLE);
  counter_configure(26, EXE_PRELOAD_HAZ_CYCLE);
  counter_configure(27, EXE_OVERLAP_HAZ_CYCLE);

  counter_configure(28, SCRATCHPAD_A_WAIT_CYCLE);
  counter_configure(29, SCRATCHPAD_B_WAIT_CYCLE);
  counter_configure(30, SCRATCHPAD_D_WAIT_CYCLE);

  counter_configure(31, ACC_A_WAIT_CYCLE);
  counter_configure(32, ACC_B_WAIT_CYCLE);
  counter_configure(33, ACC_D_WAIT_CYCLE);

  counter_configure(34, A_GARBAGE_CYCLES);
  counter_configure(35, B_GARBAGE_CYCLES);
  counter_configure(36, D_GARBAGE_CYCLES);

  counter_configure(37, IM2COL_MEM_CYCLES);
  counter_configure(38, IM2COL_ACTIVE_CYCLES);
  counter_configure(39, IM2COL_TRANSPOSER_WAIT_CYCLE);

  counter_configure(40, RESERVATION_STATION_FULL_CYCLES);
  counter_configure(41, RESERVATION_STATION_ACTIVE_CYCLES);

  counter_configure(42, LOOP_MATMUL_ACTIVE_CYCLES);
  counter_configure(43, TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES);

  counter_configure(44, RESERVATION_STATION_LD_COUNT);
  counter_configure(45, RESERVATION_STATION_ST_COUNT);
  counter_configure(46, RESERVATION_STATION_EX_COUNT);

  counter_configure(47, RDMA_BYTES_REC);
  counter_configure(48, WDMA_BYTES_SENT);

  counter_configure(49, RDMA_TOTAL_LATENCY);
  counter_configure(50, WDMA_TOTAL_LATENCY);

  for (int activation = AINIT; activation <= RELU; ++activation) {
#ifdef ACC_SCALE_T_IS_FLOAT
    for (acc_scale_t scale = 0; scale <= 1.5; scale += 0.5) {
#else
    for (acc_scale_t scale = SINIT; scale <= 12; scale += 4) {
#endif

      printf("activation: %d, scale: %d\n", activation, scale);

#ifdef ACC_SCALE_T_IS_FLOAT
	printf("ACC_SCALE_T_IS_FLOAT\n");
#else
	printf("NO ACC_SCALE_T_IS_FLOAT\n");
#endif

      static elem_t A[N][DIM][DIM] row_align(1);
      static elem_t B[N][DIM][DIM] row_align(1);
      static elem_t D[N][DIM][DIM] row_align(1);

      // We will try out every combination of A, B, D possible
      static elem_t C[N*N*N][DIM][DIM] row_align(1);
      static full_t gold_full[N*N*N][DIM][DIM];
      static elem_t gold[N*N*N][DIM][DIM];

      // ...taking into account whether we preload new weights or re-use the old ones
      static int preload[N*N*N] = {1};
      for (int i = 1; i < N*N*N; ++i)
        preload[i] = rand() % 2;

      // ...whether we pass in a D or just use zeros
      static int add_to_zeros[N*N*N];
      for (int i = 0; i < N*N*N; ++i)
        add_to_zeros[i] = rand() % 2;

      // ...and whether we accumulate on top of the previous result
      static int accumulate[N*N*N] = {0};
      for (int i = 1; i < N*N*N; ++i)
        accumulate[i] = rand() % 2;

      static int no_output[N*N*N];
      for (int i = 0; i < N*N*N-1; ++i)
        no_output[i] = accumulate[i+1];
      no_output[N*N*N-1] = 0;

      // Print the sequence out
      printf("Preloads: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", preload[i]);
      printf("\n");
      printf("Zeros: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", add_to_zeros[i]);
      printf("\n");
      printf("Accumulates: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", accumulate[i]);
      printf("\n");
      printf("No outputs: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", no_output[i]);
      printf("\n");

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = (rand() % 64) - 32;
            B[n][i][j] = (rand() % 64) - 32;
            D[n][i][j] = (rand() % 64) - 32;
          }
        }
      }

      for (size_t g = 0; g < N*N*N; ++g) {
        int a, b, d;
        operands(g, &a, &b, &d);

        // We need to find the last B value in case we aren't preloading new weights
        for (int last_g = g; last_g >= 0; --last_g) {
            int tmp_a, tmp_d;
            if (preload[last_g]) {
                operands(last_g, &tmp_a, &b, &tmp_d);
                break;
            }
        }

        if (add_to_zeros[g])
          matmul(A[a], B[b], ZERO, gold_full[g]);
        else
          matmul(A[a], B[b], D[d], gold_full[g]);

        if (accumulate[g])
          matadd(gold_full[g], gold_full[g-1], gold_full[g]);
      }

      for (size_t g = 0; g < N*N*N; ++g) {
        matscale(gold_full[g], gold[g], scale);
        if (activation == RELU)
          matrelu(gold[g], gold[g]);
      }

      uint32_t A_addr = 0;
      uint32_t B_addr = N*DIM;
      uint32_t D_addr = 2*N*DIM;
      uint32_t C_addr_acc = 1 << (ADDR_LEN-1);

      // Calculate the proper destination addresses of everything
      uint32_t C_addrs[N*N*N];
      for (size_t c = 0; c < N*N*N; ++c)
        C_addrs[c] = C_addr_acc + c*DIM;
      for (size_t c = 0; c < N*N*N; ++c) {
        int last_c;
        for (last_c = c; last_c >= 0; --last_c)
          if (!accumulate[last_c])
            break;
        if (c != last_c)
          C_addrs[c] = C_addrs[last_c] | (1 << (ADDR_LEN-2));
      }

      // printf("Moving in\n");
      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(A[n], A_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(B[n], B_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        if (n == N-1) {
          gemmini_mvin(D[n], D_addr + n*DIM);
        } else {
          gemmini_mvin(D[n], D_addr + n*DIM);
        }

      // printf("Setting mode\n");
      gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
      gemmini_extended_config_st(DIM * sizeof(elem_t), activation, scale);

      counter_snapshot_take();

      counter_val[44] = counter_read(44);
      counter_val[45] = counter_read(45);
      counter_val[46] = counter_read(46);
      counter_val[47] = counter_read(47);
      counter_val[48] = counter_read(48);
      counter_val[49] = counter_read(49);
      counter_val[50] = counter_read(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

      // printf("Matmulling\n");
      for (size_t c = 0; c < N*N*N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);

        uint32_t d_addr = D_addr + d*DIM;
        if (add_to_zeros[c])
          d_addr = GARBAGE_ADDR;

        if (!preload[c]) {
          gemmini_preload_zeros(C_addrs[c]);
          gemmini_compute_accumulated(A_addr + a*DIM, d_addr);
        } else {
          gemmini_preload(B_addr + b*DIM, C_addrs[c]);
          gemmini_compute_preloaded(A_addr + a*DIM, d_addr);
        }
      }

      counter_snapshot_take();

      counter_val[44] = counter_read(44);
      counter_val[45] = counter_read(45);
      counter_val[46] = counter_read(46);
      counter_val[47] = counter_read(47);
      counter_val[48] = counter_read(48);
      counter_val[49] = counter_read(49);
      counter_val[50] = counter_read(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

      // printf("Moving out\n");
      for (size_t c = 0; c < N*N*N; ++c)
        if (!no_output[c]) {
          gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN-2)));
        }

      counter_snapshot_take();

      counter_val[44] = counter_read(44);
      counter_val[45] = counter_read(45);
      counter_val[46] = counter_read(46);
      counter_val[47] = counter_read(47);
      counter_val[48] = counter_read(48);
      counter_val[49] = counter_read(49);
      counter_val[50] = counter_read(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

      gemmini_fence();

      // for (int i = 0; i <= 44; i++) {
    	// 	counter_vals[i] = counter_read(i);
	// }

        //want to avoid using unnessesary branch logic, or incrementing i, here etc...
	counter_val[0] = counter_read(0);
	counter_val[1] = counter_read(1);
	counter_val[2] = counter_read(2);
	counter_val[3] = counter_read(3);
	counter_val[4] = counter_read(4);
	counter_val[5] = counter_read(5);
	counter_val[6] = counter_read(6);
	counter_val[7] = counter_read(7);
	counter_val[8] = counter_read(8);
	counter_val[9] = counter_read(9);
	counter_val[10] = counter_read(10);
	counter_val[11] = counter_read(11);
	counter_val[12] = counter_read(12);
	counter_val[13] = counter_read(13);
	counter_val[14] = counter_read(14);
	counter_val[15] = counter_read(15);
	counter_val[16] = counter_read(16);
	counter_val[17] = counter_read(17);
	counter_val[18] = counter_read(18);
	counter_val[19] = counter_read(19);
	counter_val[20] = counter_read(20);
	counter_val[21] = counter_read(21);
	counter_val[22] = counter_read(22);
	counter_val[23] = counter_read(23);
	counter_val[24] = counter_read(24);
	counter_val[25] = counter_read(25);
	counter_val[26] = counter_read(26);
	counter_val[27] = counter_read(27);
	counter_val[28] = counter_read(28);
	counter_val[29] = counter_read(29);
	counter_val[30] = counter_read(30);
	counter_val[31] = counter_read(31);
	counter_val[32] = counter_read(32);
	counter_val[33] = counter_read(33);
	counter_val[34] = counter_read(34);
	counter_val[35] = counter_read(35);
	counter_val[36] = counter_read(36);
	counter_val[37] = counter_read(37);
	counter_val[38] = counter_read(38);
	counter_val[39] = counter_read(39);
	counter_val[40] = counter_read(40);
	counter_val[41] = counter_read(41);
	counter_val[42] = counter_read(42);
	counter_val[43] = counter_read(43);
	counter_val[44] = counter_read(44);
	counter_val[45] = counter_read(45);
	counter_val[46] = counter_read(46);
	counter_val[47] = counter_read(47);
	counter_val[48] = counter_read(48);
	counter_val[49] = counter_read(49);
	counter_val[50] = counter_read(50);
	

	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i <= 50; ++i) {
		printf("%s, %d\n", counter_names[i], counter_val[i]);
	}

	printf("\n\n\n");

	counter_reset();

      printf("Moved out\n");
      for (int n = 0; n < N*N*N; ++n) {
        if (!no_output[n]) {
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");
        }
      }	

      // printf("Checking\n");
      for (int n = 0; n < N*N*N; ++n)
        if (!no_output[n] && !is_equal(C[n], gold[n])) {
          printf("activation: %d, scale: %d\n", activation, scale);
          printf("Actual (%d):\n", n);
          printMatrix(C[n]);
          printf("\nGold:\n");
          printMatrix(gold[n]);
          exit(1);
        }
    }
  }

  exit(0);
}

