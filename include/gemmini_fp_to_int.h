// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_FP_TO_INT_H
#define SRC_MAIN_C_GEMMINI_FP_TO_INT_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"
#include "include/gemmini_32FP_params.h"

#define GEMMINI_ASSERTIONS

// Accelerator interface
#include "rocc-software/src/xcustom.h"

// Counter Definition
#include "include/gemmini_counter.h"

#ifndef EXPOSE_TOP_LEVEL_FNS
#define _STATIC static
#else
#define _STATIC
#endif

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

#define k_MVIN3 14

#define k_COUNTER 126

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

// CLKGATE_EN: 22
#define k_MVOUT_SPAD 23
#define k_LOOP_WS_CONFIG_SPAD_AB 24
#define k_LOOP_WS_CONFIG_SPAD_C 25

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct) //this worked with the name unchanged for gemmini_fp.h


// mvin and mvout
#define fp_to_int_gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define fp_to_int_gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define fp_to_int_gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define fp_to_int_gemmini_block_mvin(dram_addr, spad_addr, len) \
  fp_to_int_gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define fp_to_int_gemmini_mvin(dram_addr, spad_addr) \
  fp_to_int_gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define fp_to_int_gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define fp_to_int_gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

#define fp_to_int_gemmini_mvout_spad(dst_addr, src_addr) \
  fp_to_int_gemmini_extended_mvout_spad(dst_addr, 1, src_addr, DIM, DIM)

#define fp_to_int_gemmini_mvout(dram_addr, spad_addr) \
  fp_to_int_gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define fp_to_int_gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define fp_to_int_gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define fp_to_int_gemmini_compute_preloaded(A, BD) \
  fp_to_int_gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define fp_to_int_gemmini_compute_accumulated(A, BD) \
  fp_to_int_gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define fp_to_int_gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define fp_to_int_gemmini_preload(BD, C) \
  fp_to_int_gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define fp_to_int_gemmini_preload_zeros(C) \
  fp_to_int_gemmini_preload(GARBAGE_ADDR, C)

// config
#define fp_to_int_gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG); \

#define fp_to_int_gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  fp_to_int_gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define fp_to_int_gemmini_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  fp_to_int_gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define fp_to_int_gemmini_config_ex(dataflow, sys_act, sys_shift) \
    fp_to_int_gemmini_extended_config_ex(dataflow, sys_act, sys_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define fp_to_int_gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define fp_to_int_gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  fp_to_int_gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

#define fp_to_int_gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  fp_to_int_gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

#define fp_to_int_gemmini_extended2_config_ld(stride, scale, shrunk) \
  fp_to_int_gemmini_extended3_config_ld(stride, scale, shrunk, 0)

#define fp_to_int_gemmini_extended_config_ld(stride, scale) \
  fp_to_int_gemmini_extended2_config_ld(stride, scale, false)

#define fp_to_int_gemmini_config_ld(stride) \
  fp_to_int_gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define fp_to_int_gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define fp_to_int_gemmini_extended_config_st(stride, acc_act, acc_scale) \
    fp_to_int_gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define fp_to_int_gemmini_config_st(stride) \
    fp_to_int_gemmini_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

#define fp_to_int_gemmini_config_norm(q_const, q_const_type, set_stats_id_only, act_msb, stat_id, igelu_qb, igelu_qc) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (((uint64_t) ((uint32_t) q_const)) << 32) | ((q_const_type & 1) << 18) | ((set_stats_id_only & 1) << 17) | ((act_msb & 1) << 16) | ((uint64_t)stat_id << 8) | CONFIG_BERT, ((uint64_t)((uint32_t)(igelu_qc)) << 32) | ((uint64_t)((uint32_t)(igelu_qb))), k_CONFIG)

// flush
#define fp_to_int_gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define fp_to_int_gemmini_fence() asm volatile("fence")




// weight-stationary matmul loop
#define fp_to_int_gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

enum fp_to_int_tiled_matmul_type_t {FP_TO_INT_OS, FP_TO_INT_WS, FP_TO_INT_CPU};

static void fp_to_int_sp_tiled_matmul_ws(const void * A, const void * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

  // Combined loop
  fp_to_int_gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
    A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
    a_transpose, b_transpose,
    full_C, low_D, !no_bias || D == NULL,
    act, a_spad_id, b_spad_id, false);
}

static void fp_to_int_tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const void* A, const void* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  size_t A_element_size = (A_scale_factor != MVIN_SCALE_IDENTITY) ? sizeof(fp_elem_t) : sizeof(elem_t);
  size_t B_element_size = (B_scale_factor != MVIN_SCALE_IDENTITY) ? sizeof(fp_elem_t) : sizeof(elem_t);



  fp_to_int_gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  fp_to_int_gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  fp_to_int_gemmini_extended3_config_ld(stride_A * A_element_size, A_scale_factor, false, 0);
  fp_to_int_gemmini_extended3_config_ld(stride_B * B_element_size, B_scale_factor, false, 1)
  fp_to_int_gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S*S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;

    fp_to_int_gemmini_config_norm(0, 0, 0, 0, 0, qb, qc);
  }

  if (act == SOFTMAX) {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int) (0.693147 / bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / bert_scale;
    const acc_t qc = c / (a*bert_scale*bert_scale);

    fp_to_int_gemmini_config_norm(qln2, 0, 0, 1, 0, qb, qc);
    fp_to_int_gemmini_config_norm(qln2_inv, 1, 0, 1, 0, qb, qc);
  }

  void (*inner)(const void *, const void *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        int, int, int);

    inner = &fp_to_int_sp_tiled_matmul_ws;

  // reuse operand if it fits scratchpad
  int a_spad_id = 0;
  int b_spad_id = 0;
  bool b_reuse = (J0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
  bool a_reuse = (I0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((j0+k0) == 0) ? 1 : 2;

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

	const elem_t * A_int = (const elem_t *) A;
	const elem_t * B_int = (const elem_t *) B;
	const fp_elem_t * A_float = (const fp_elem_t *) A;
	const fp_elem_t * B_float = (const fp_elem_t *) B;

	const elem_t * a_int;
        const elem_t * b_int;
	
	const fp_elem_t * a_float;
        const fp_elem_t * b_float;

	const void * a_void;
	const void * b_void;

	if ((A_scale_factor == MVIN_SCALE_IDENTITY) && (B_scale_factor == MVIN_SCALE_IDENTITY)) {
		a_int = a_transpose ? (A_int + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A_int + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);
        	b_int = b_transpose ? (B_int + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B_int + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

		a_void = (const void *) a_int;
		b_void = (const void *) b_int;

	} else if ((A_scale_factor == MVIN_SCALE_IDENTITY) && (B_scale_factor != MVIN_SCALE_IDENTITY)) {
		a_int = a_transpose ? (A_int + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A_int + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);
        	b_float = b_transpose ? (B_float + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B_float + k0*tile_K*DIM*stride_B + j0*tile_J*DIM); //don't think I need to multiply by sizeof(fc_elem_t) for this

		a_void = (const void *) a_int;
		b_void = (const void *) b_float;

		stride_B = sizeof(fp_elem_t) * stride_B; 

	  
	} else if ((A_scale_factor != MVIN_SCALE_IDENTITY) && (B_scale_factor == MVIN_SCALE_IDENTITY)) {
		a_float = a_transpose ? (A_float + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A_float + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);
        	b_int = b_transpose ? (B_int + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B_int + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

		stride_A = sizeof(fp_elem_t) * stride_A; 

		a_void = (const void *) a_float;
		b_void = (const void *) b_int;
	} else if ((A_scale_factor != MVIN_SCALE_IDENTITY) && (B_scale_factor != MVIN_SCALE_IDENTITY)) {
		a_float = a_transpose ? (A_float + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A_float + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);
        	b_float = b_transpose ? (B_float + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B_float + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

		stride_A = sizeof(fp_elem_t) * stride_A; 
		stride_B = sizeof(fp_elem_t) * stride_B; 

		a_void = (const void *) a_float;
		b_void = (const void *) b_float;
	}

        if(a_reuse && j0 >= 1) a_void = NULL;
        if(b_reuse && i0 >= 1) b_void = NULL;
        //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        (*inner)(a_void, b_void, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act, a_spad_id, b_spad_id);
      }

  fp_to_int_gemmini_fence();
}

static void fp_to_int_tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const void* A, const void* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum fp_to_int_tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == FP_TO_INT_WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][14] = {"FP_TO_INT_OS", "FP_TO_INT_WS", "FP_TO_INT_CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == FP_TO_INT_OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == FP_TO_INT_WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == FP_TO_INT_CPU && (full_C || low_D)) ||
      (tiled_matmul_type == FP_TO_INT_OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX) {
    if (tiled_matmul_type == FP_TO_INT_OS) {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J) {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
//   if (tiled_matmul_type == FP_TO_INT_OS || tiled_matmul_type == FP_TO_INT_WS) {
    fp_to_int_tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, bert_scale, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type);
//     }
}

static size_t fp_to_int_tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

static size_t fp_to_int_tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}


_STATIC void fp_to_int_tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const void* A, const void* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum fp_to_int_tiled_matmul_type_t tiled_matmul_type) {

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = tiled_matmul_type == FP_TO_INT_WS;

    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX) {
       tile_I = 1;
       tile_J = dim_J_padded/DIM;
       tile_K = 1;
    } else if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (fp_to_int_tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          fp_to_int_tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (fp_to_int_tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          fp_to_int_tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (fp_to_int_tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = fp_to_int_tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = fp_to_int_tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    exit(EXIT_SUCCESS);
#endif
#endif

    fp_to_int_tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_FP_TO_INT_H

