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

#if 3*N*DIM > (BANK_NUM * BANK_ROWS) || N*N*N*DIM > ACC_ROWS
//#error scratchpad or accumulator not big enough
#endif

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

int counter_val[51];

  

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

	
	int activation = RELU;
	acc_scale_t scale = 1;

      for (int array_DIM = 1; array_DIM <= 32; ++array_DIM) {
	printf("array_DIM: %d\n\n", array_DIM);
	gemmini_flush(0);
  	gemmini_config_ld(array_DIM * sizeof(elem_t));

	elem_t A[array_DIM][array_DIM];
        elem_t B[array_DIM][array_DIM];
        elem_t D[array_DIM][array_DIM];

        elem_t C[array_DIM][array_DIM];
        //full_t gold_full[array_DIM][array_DIM];
        //elem_t gold[array_DIM][array_DIM];

        for (size_t i = 0; i < array_DIM; ++i) {
          for (size_t j = 0; j < array_DIM; ++j) {
            A[i][j] = (rand() % 4);
            B[i][j] = (rand() % 4);
            D[i][j] = (rand() % 4);
          }
        }

        // //matmul
	// for (size_t r = 0; r < array_DIM; ++r) {
	// 	for (size_t c = 0; c < array_DIM; ++c) {
	// 		gold_full[r][c] = D[r][c];
	// 		for (size_t k = 0; k < array_DIM; ++k) {
	// 			gold_full[r][c] += A[r][k]*B[k][c];
	// 		}
	// 	}
	// }
        // //matscale
	// for (size_t r = 0; r < array_DIM; ++r) {
	// 	for (size_t c = 0; c < array_DIM; ++c) {
	// 	// Bitshift and round element
	// 	full_t scaled = ACC_SCALE(gold_full[r][c], scale);
	// 		#ifndef ELEM_T_IS_FLOAT
	// 			full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
	// 			gold[r][c] = elem;
	// 		#else
	// 			gold[r][c] = scaled;
	// 		#endif
	// 	}
	// }

        // //matrelu
	// for (size_t r = 0; r < array_DIM; ++r) {
    	// for (size_t c = 0; c < array_DIM; ++c) {
      	// 	gold[r][c] = gold[r][c] > 0 ? gold[r][c] : 0;
	// 	}
	// }

      uint32_t A_addr = 0;
      uint32_t B_addr = DIM;
      uint32_t D_addr = 2*DIM;
      uint32_t C_addr = 3*DIM;

      // Calculate the proper destination addresses of everything

    printf("A:\n");
	for (size_t i = 0; i < array_DIM; ++i) {
		for (size_t j = 0; j < array_DIM; ++j) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("B:\n");
	for (size_t i = 0; i < array_DIM; ++i) {
		for (size_t j = 0; j < array_DIM; ++j) {
			printf("%d ", B[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("D:\n");
	for (size_t i = 0; i < array_DIM; ++i) {
		for (size_t j = 0; j < array_DIM; ++j) {
			printf("%d ", D[i][j]);
		}
		printf("\n");
	}
	printf("\n");


	gemmini_config_ld(array_DIM * sizeof(elem_t));
	gemmini_extended_mvin(A, A_addr, array_DIM, array_DIM);
	gemmini_config_ld(array_DIM * sizeof(elem_t));
	gemmini_extended_mvin(B, B_addr, array_DIM, array_DIM);
	gemmini_config_ld(array_DIM * sizeof(elem_t));
	gemmini_extended_mvin(D, D_addr, array_DIM, array_DIM);


      gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
      gemmini_extended_config_st(array_DIM * sizeof(elem_t), activation, scale);

      counter_snapshot_take();

      counter_val[44] = counter_read_extra(44);
      counter_val[45] = counter_read_extra(45);
      counter_val[46] = counter_read_extra(46);
      counter_val[47] = counter_read_extra(47);
      counter_val[48] = counter_read_extra(48);
      counter_val[49] = counter_read_extra(49);
      counter_val[50] = counter_read_extra(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

        //uint32_t d_addr = D_addr + d*DIM;

        gemmini_extended_preload(B_addr, C_addr, array_DIM, array_DIM, array_DIM, array_DIM);
        gemmini_extended_compute_preloaded(A_addr, D_addr, array_DIM, array_DIM, array_DIM, array_DIM);

      counter_snapshot_take();

      counter_val[44] = counter_read_extra(44);
      counter_val[45] = counter_read_extra(45);
      counter_val[46] = counter_read_extra(46);
      counter_val[47] = counter_read_extra(47);
      counter_val[48] = counter_read_extra(48);
      counter_val[49] = counter_read_extra(49);
      counter_val[50] = counter_read_extra(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

      gemmini_extended_mvout(C, C_addr, array_DIM, array_DIM);

      counter_snapshot_take();

      counter_val[44] = counter_read_extra(44);
      counter_val[45] = counter_read_extra(45);
      counter_val[46] = counter_read_extra(46);
      counter_val[47] = counter_read_extra(47);
      counter_val[48] = counter_read_extra(48);
      counter_val[49] = counter_read_extra(49);
      counter_val[50] = counter_read_extra(50);

      for (int i = 44; i <= 50; ++i) {
	printf("%s, %d\n", counter_names[i], counter_val[i]);
      }

      printf("\n\n");

      counter_snapshot_reset();

      gemmini_fence();

        //want to avoid using unnessesary branch logic, or incrementing i, here etc...
	counter_val[0] = counter_read_extra(0);
	counter_val[1] = counter_read_extra(1);
	counter_val[2] = counter_read_extra(2);
	counter_val[3] = counter_read_extra(3);
	counter_val[4] = counter_read_extra(4);
	counter_val[5] = counter_read_extra(5);
	counter_val[6] = counter_read_extra(6);
	counter_val[7] = counter_read_extra(7);
	counter_val[8] = counter_read_extra(8);
	counter_val[9] = counter_read_extra(9);
	counter_val[10] = counter_read_extra(10);
	counter_val[11] = counter_read_extra(11);
	counter_val[12] = counter_read_extra(12);
	counter_val[13] = counter_read_extra(13);
	counter_val[14] = counter_read_extra(14);
	counter_val[15] = counter_read_extra(15);
	counter_val[16] = counter_read_extra(16);
	counter_val[17] = counter_read_extra(17);
	counter_val[18] = counter_read_extra(18);
	counter_val[19] = counter_read_extra(19);
	counter_val[20] = counter_read_extra(20);
	counter_val[21] = counter_read_extra(21);
	counter_val[22] = counter_read_extra(22);
	counter_val[23] = counter_read_extra(23);
	counter_val[24] = counter_read_extra(24);
	counter_val[25] = counter_read_extra(25);
	counter_val[26] = counter_read_extra(26);
	counter_val[27] = counter_read_extra(27);
	counter_val[28] = counter_read_extra(28);
	counter_val[29] = counter_read_extra(29);
	counter_val[30] = counter_read_extra(30);
	counter_val[31] = counter_read_extra(31);
	counter_val[32] = counter_read_extra(32);
	counter_val[33] = counter_read_extra(33);
	counter_val[34] = counter_read_extra(34);
	counter_val[35] = counter_read_extra(35);
	counter_val[36] = counter_read_extra(36);
	counter_val[37] = counter_read_extra(37);
	counter_val[38] = counter_read_extra(38);
	counter_val[39] = counter_read_extra(39);
	counter_val[40] = counter_read_extra(40);
	counter_val[41] = counter_read_extra(41);
	counter_val[42] = counter_read_extra(42);
	counter_val[43] = counter_read_extra(43);
	counter_val[44] = counter_read_extra(44);
	counter_val[45] = counter_read_extra(45);
	counter_val[46] = counter_read_extra(46);
	counter_val[47] = counter_read_extra(47);
	counter_val[48] = counter_read_extra(48);
	counter_val[49] = counter_read_extra(49);
	counter_val[50] = counter_read_extra(50);
	

	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i <= 50; ++i) {
		printf("%s, %d\n", counter_names[i], counter_val[i]);
	}

	printf("\n\n\n");

	counter_reset();

    printf("Moved out\n");
	printf("C:\n");
	for (size_t i = 0; i < array_DIM; ++i) {
		for (size_t j = 0; j < array_DIM; ++j) {
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}
    }
  }