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

// #ifndef array_dim
// #define array_dim DIM
// #endif
#define array_dim 8
// normal counter funcitons only works if you stick to the default number of counters (8)
static void counter_configure_extra(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0x3f) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

static uint32_t counter_read_extra(size_t index) {
  uint32_t config_reg = (index & 0x3f) << 4;
  uint32_t res;
  gemmini_counter_access(res, config_reg);
  return res;
}

static uint16_t float_to_uint16(float f) {
	union {
		float f;
		uint32_t b;
	} un;

	un.f = f;
	uint32_t bits = un.b;

	uint16_t sign = (bits >> 31) & 1;
	uint16_t exponent = ((bits >> 23) & 0xff) - 127; //full exponent
	uint32_t mantissa = bits & 0x7fffff; //full mantissa, 23 bits

	uint16_t round_up = 0;

	uint16_t mantissa_last_bit = ((mantissa & (1 << 13)) >> 13);
	uint16_t guard = ((mantissa & (1 << 12)) >> 12);
	uint16_t round = ((mantissa & (1 << 11)) >> 11);
	//uint16_t sticky = ((mantissa & (1 << 10)) >> 10);
	uint16_t sticky = (mantissa & 0x7ff) ? 1 : 0;

	round_up = guard && (round || sticky);

	if (guard && !round && !sticky) {
		round_up = mantissa_last_bit; //if mantissa last bit is 1 round up, if 0 don't round
	}

	uint16_t result = (sign << 15) | (((exponent & 0x1f) + 15) << 10) | (((mantissa >> 13) + round_up) & 0x3ff);
	return result;
}

static uint16_t float_to_brain16(float f) {
	union {
		float f;
		uint32_t b;
	} un;

	un.f = f;
	uint32_t bits = un.b;

	uint16_t sign = (bits >> 31) & 1;
	//uint16_t exponent = ((bits >> 23) & 0xff) - 127; //full exponent
	uint16_t exponent = ((bits >> 23) & 0xff); //full exponent
	uint32_t mantissa = bits & 0x7fffff; //full mantissa, 23 bits

	uint16_t round_up = 0;

	uint16_t mantissa_last_bit = ((mantissa & (1 << 16)) >> 16);
	uint16_t guard = ((mantissa & (1 << 15)) >> 15);
	uint16_t round = ((mantissa & (1 << 14)) >> 14);
	//uint16_t sticky = ((mantissa & (1 << 10)) >> 10);
	uint16_t sticky = (mantissa & 0x3fff) ? 1 : 0;

	round_up = guard && (round || sticky);

	if (guard && !round && !sticky) {
		round_up = mantissa_last_bit; //if mantissa last bit is 1 round up, if 0 don't round
	}

	uint16_t result = (sign << 15) | (exponent << 7) | (((mantissa >> 16) + round_up) & 0x7f);
	return result;
}

int is_equal_array_dim(elem_t x[array_dim][array_dim], elem_t y[array_dim][array_dim]) {//normal is_equal function expects array dim diamentions
    for (size_t i = 0; i < array_dim; i++) {
        for (size_t j = 0; j < array_dim; j++) {
#ifndef ELEM_T_IS_FLOAT
            if (x[i][j] != y[i][j]) return 0;
#else
            bool isnanx = elem_t_isnan(x[i][j]);
            bool isnany = elem_t_isnan(y[i][j]);
            if (x[i][j] != y[i][j] && !(isnanx && isnany)) return 0;
#endif
	}
    }
    return 1;
}

// static void print_float(elem_t float) { NO WAY TO DO AN EXPONENTIAL IN BAREMETAL
// 	elem_t_bits bits = elem_t_to_elem_t_bits(float);

// 	elem_t_bits sign = (bits >> 31) & 1;
// 	elem_t_bits exp = ((bits >> 23) & 0xff) - 127;
// 	elem_t_bits mantissa = (bits & 0x7fffff);

// 	if (sign) {
// 		printf("-");
// 	}

// }

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

  
  for (int i = 0; i < 51; ++i) {
	counter_configure_extra(i, i + 1);
  }

	printf("array_dim: %d\n\n", array_dim);
	gemmini_flush(0);

	counter_reset();

  	gemmini_config_ld(array_dim * sizeof(elem_t));

	elem_t A[array_dim][array_dim];
        elem_t B[array_dim][array_dim];
        elem_t D[array_dim][array_dim];

        acc_t C[array_dim][array_dim]; //output full_c
	acc_t gold[array_dim][array_dim];

        for (size_t i = 0; i < array_dim; ++i) {
          for (size_t j = 0; j < array_dim; ++j) {
            A[i][j] = float_to_brain16((float) (rand_double() / 32));
            B[i][j] = float_to_brain16((float) (rand_double() / 32));
            D[i][j] = float_to_brain16((float) (rand_double() / 32));
          }
        }

//       uint32_t A_addr = 0;
//       uint32_t B_addr = DIM;
//       uint32_t D_addr = 2*DIM;
//       uint32_t C_addr = 3*DIM;

      // Calculate the proper destination addresses of everything

    printf("A:\n"); //NO WAY TO PRINT FLOATS IN A READABLE WAY EASILY
	for (size_t i = 0; i < array_dim; ++i) {
		for (size_t j = 0; j < array_dim; ++j) {
			printf("%x ", elem_t_to_elem_t_bits(A[i][j]));
		}
		printf("\n");
	}
	printf("\n");

	printf("B:\n");
	for (size_t i = 0; i < array_dim; ++i) {
		for (size_t j = 0; j < array_dim; ++j) {
			printf("%x ", elem_t_to_elem_t_bits(B[i][j]));
		}
		printf("\n");
	}
	printf("\n");

	printf("D:\n");
	for (size_t i = 0; i < array_dim; ++i) {
		for (size_t j = 0; j < array_dim; ++j) {
			printf("%x ", elem_t_to_elem_t_bits(D[i][j]));
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
	tiled_matmul_auto(array_dim, array_dim, array_dim,
            (elem_t*)A, (elem_t*)B,
	    &D[0][0], (elem_t*)C,
            array_dim, array_dim, array_dim, array_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0, 
	    false,
            0, 0,
            true, false,
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
    const int total_macs = array_dim * array_dim * array_dim;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n\n\n", utilization);
        //

    for (int i = 0; i < 51; ++i) { //assuming the compiler will unroll the loop
	counter_val_end[i] = counter_read_extra(i);
    }
	
	//printf("Counters after compute: EXE_ACTIVE_CYCLE: %d, EXE_FLUSH_CYCLE: %d, EXE_CONTROL_Q_BLOCK_CYCLE: %d, EXE_PRELOAD_HAZ_CYCLE: %d, EXE_OVERLAP_HAZ_CYCLE: %d, MAIN_EX_CYCLES: %d\n", counter_val_0, counter_val_1, counter_val_2, counter_val_3, counter_val_4, counter_val_5);

	for (int i = 0; i <= 50; ++i) {
		printf("DIM %d: %s, %d\n", array_dim, counter_names[i], counter_val_end[i] - counter_val_start[i]);
	}

	printf("\n\n\n");

	//counter_reset();

      printf("Moved out\n");
        //   printf("C:\n");
	//   for (size_t i = 0; i < array_dim; ++i) {
    	//     for (size_t j = 0; j < array_dim; ++j) {
      	//       printf("%f ", C[i][j]);
	//     }
    	//     printf("\n");
  	//   }
        //   printf("\n");

	tiled_matmul_auto(array_dim, array_dim, array_dim,
            (elem_t*)A, (elem_t*)B,
	    &D[0][0], (elem_t*)gold,
            array_dim, array_dim, array_dim, array_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0, 
	    false,
            0, 0,
            true, false,
            0,
            CPU); 

	if (!is_equal_array_dim(C, gold)) {
          printf("C != GOLD\n");
        } else {
	  printf("C == GOLD\n");
	}

	printf("C:\n");
        //printMatrix(C[n]);
	for (size_t i = 0; i < array_dim; ++i) {
		for (size_t j = 0; j < array_dim; ++j) {
			printf("%x ", elem_t_to_elem_t_bits(C[i][j]));
		}
		printf("\n");
	}
	printf("\n");
        printf("\nGold:\n");
	for (size_t i = 0; i < array_dim; ++i) {
		for (size_t j = 0; j < array_dim; ++j) {
			printf("%x ", elem_t_to_elem_t_bits(gold[i][j]));
		}
		printf("\n");
	}
	printf("\n");
        //printMatrix(gold[n]);

	exit(0);
  }