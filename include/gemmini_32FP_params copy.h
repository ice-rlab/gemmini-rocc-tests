#ifndef GEMMINI_PARAMS_FP32_H
#define GEMMINI_PARAMS_FP32_H

#include <stdint.h>
#include <limits.h>

#define FP_XCUSTOM_ACC 2
#define FP_DIM 4
#define FP_ADDR_LEN 32
#define FP_BANK_NUM 4
#define FP_BANK_ROWS 4096
#define FP_ACC_ROWS 4096
#define FP_MAX_BYTES 64
#define FP_MAX_BLOCK_LEN (FP_MAX_BYTES/(FP_DIM*4))
#define FP_MAX_BLOCK_LEN_ACC (FP_MAX_BYTES/(FP_DIM*4))

typedef float fp_elem_t;
static const fp_elem_t fp_elem_t_max = 3.4028235E38;
static const fp_elem_t fp_elem_t_min = -3.4028235E38;
typedef float fp_acc_t;
typedef double fp_full_t;

#define FP_ELEM_T_IS_FLOAT
#define FP_ELEM_T_EXP_BITS 8
#define FP_ELEM_T_SIG_BITS 24
#define FP_ACC_T_EXP_BITS 8
#define FP_ACC_T_SIG_BITS 24
typedef uint32_t fp_elem_t_bits;
typedef uint32_t fp_acc_t_bits;

#define FP_HAS_MVIN_SCALE
typedef float fp_scale_t;
typedef uint32_t fp_scale_t_bits;

#define FP_HAS_MVIN_ACC_SCALE
typedef float fp_scale_acc_t;
typedef uint32_t fp_scale_acc_t_bits;

typedef float fp_acc_scale_t;
typedef uint32_t fp_acc_scale_t_bits;

#define FP_row_align(blocks) __attribute__((aligned(blocks*FP_DIM*sizeof(fp_elem_t))))
#define FP_row_align_acc(blocks) __attribute__((aligned(blocks*FP_DIM*sizeof(fp_acc_t))))

#define FP_MVIN_SCALE_IDENTITY 1.0

#define FP_ACC_SCALE_IDENTITY 1.0

#define FP_ROUNDING_RIGHT_SHIFT(x, shift) \
    ((x) / (1 << (shift)))

#ifdef __cplusplus
#define FP_SAME_TYPE(x) decltype(x)
#else
#define FP_SAME_TYPE(x) typeof(x)
#endif

#define FP_ROUND_NEAR_EVEN(x) \
    ({ const FP_SAME_TYPE(x) x_ = (x); \
         const long long i = x_; \
         const long long next = x_ < 0 ? x_ - 1 : x_ + 1; \
         FP_SAME_TYPE(x) rem = x_ - i; \
         rem = rem < 0 ? -rem : rem; \
         FP_SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( \
                     i % 2 == 0 ? i : next)); \
         result; })

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define FP_ROUNDING_RIGHT_SHIFT_BITS(x, shift) \
((shift) > 0 ? (((x) >> (shift)) + \
    (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
         ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#define FP_ACC_SCALE(x, scale) \
    ((x) * (scale))

#define FP_MVIN_SCALE(x, scale) \
    ((x) * (scale))

#define FP_MVIN_SCALE_ACC(x, scale) \
    ((x) * (scale))

#define FP_ACC_SCALE_T_IS_FLOAT
#define FP_ACC_SCALE_EXP_BITS 8
#define FP_ACC_SCALE_SIG_BITS 24

#define FP_ACC_READ_SMALL_WIDTH
#define FP_ACC_READ_FULL_WIDTH

#define FP_HAS_FIRST_LAYER_OPTIMIZATIONS

#endif // GEMMINI_PARAMS_FP32_H
