#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "mobilenet_params.h"
#include "images.h"

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

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

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

for (int i = 0; i < 51; ++i) {
	counter_configure_extra(i, i + 1);
}

    gemmini_flush(0);

    counter_reset();

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv = true;
    
    if (argc < 3) {
        conv = true;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;
    
    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // conv_1
    if (!conv) {
      start = read_cycles();

        im2col(conv_1_params.batch_size, conv_1_params.in_channels,
            conv_1_params.in_row_dim, conv_1_params.in_col_dim,
            conv_1_params.I, conv_1_params.K,
            images, conv_1_in, &conv_1_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_in, conv_1_w, conv_1_b, conv_1_out,
            RELU, conv_1_params.output_scale, true,
            tiled_matmul_type, check, "conv_1");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
            conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
            conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

            RELU, conv_1_params.output_scale,
            conv_1_params.pool_size, 0, conv_1_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();

	conv_cycles += end - start;
        printf("conv_1: %llu\n", end-start);

	for (int i = 0; i < 51; ++i) { //assuming the compiler will unroll the loop
		counter_val[i] = counter_read_extra(i);
		printf("%s, %d\n", counter_names[i], counter_val[i]);
    	}


	
	// for (int i = 0; i < 32; ++i) {
	// 	for (int j = 0; j < 27; ++j) {
	// 		printf("%d ", conv_1_w[i][j]);
	// 	}
	// 	printf("\n");
	// }

	printf("\n\n\n");

        
    }

//     // conv_dw_2
//     start = read_cycles();

//     if (!conv) {
//         conv_dw_with_col2im(conv_1_params.I, conv_1_params.J, conv_dw_2_params.I, conv_dw_2_params.J,
//             conv_dw_2_params.batch_size, conv_dw_2_params.in_channels,
//             conv_dw_2_params.out_row_dim, conv_dw_2_params.out_col_dim,
//             conv_dw_2_params.kernel_size,
//             conv_1_out, conv_dw_2_w, conv_dw_2_b, conv_dw_2_out, &conv_dw_2_params);
//     } else {
//         tiled_conv_dw_auto(
//             conv_dw_2_params.batch_size, conv_dw_2_params.in_row_dim, conv_dw_2_params.in_col_dim,
//             conv_dw_2_params.in_channels,
//             conv_dw_2_params.out_row_dim, conv_dw_2_params.out_col_dim,
//             conv_dw_2_params.stride, conv_dw_2_params.padding, conv_dw_2_params.kernel_size,

//             (elem_t*)conv_1_out, (elem_t*)conv_dw_2_w, (acc_t*)conv_dw_2_b, (elem_t*)conv_dw_2_out,

//             RELU, conv_dw_2_params.output_scale,
//             conv_dw_2_params.pool_size, 0, conv_dw_2_params.pool_padding,

//             tiled_matmul_type);
//     }

//     end = read_cycles();
//     conv_dw_cycles += end - start;
//     printf("conv_dw_2: %llu \n", end - start);

//     // conv_3
//     if (!conv) {
//         start = read_cycles();

//         tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
//             conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out,
//             NO_ACTIVATION, conv_3_params.output_scale, true,
//             tiled_matmul_type, check, "conv_3");

//         end = read_cycles();
//         matmul_cycles += end - start;

//     } else {
//         start = read_cycles();

//         tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
//             conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out,
//             NO_ACTIVATION, conv_3_params.output_scale, true,
//             tiled_matmul_type, check, "conv_3");

//         end = read_cycles();
//         matmul_cycles += end - start;
//     }

//     printf("matmul_3: %llu\n", end-start);

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

    printf("PASS\n");
    exit(0);
}

