// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_TESTUTILS_FP_H
#define SRC_MAIN_C_GEMMINI_TESTUTILS_FP_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_32FP_params.h"
#include "include/gemmini_fp.h"

#ifdef BAREMETAL
#undef assert
#define assert(expr) \
    if (!(expr)) { \
      printf("Failed assertion: " #expr "\n  " __FILE__ ":%u\n", __LINE__); \
      exit(1); \
    }
#endif

// #define GEMMINI_ASSERTIONS

// Matmul utility functions
static void fp_matmul(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static void fp_matmul_short(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_elem_t C[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C[r][c] += A[r][k]*B[k][c];
    }
}

static void fp_matmul_full(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_full_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  // Identical to the other matmul function, but with a 64-bit bias
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static void fp_matmul_A_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[k][r]*B[k][c];
    }
}

static void fp_matmul_short_A_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_elem_t C[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C[r][c] += A[k][r]*B[k][c];
    }
}

static void fp_matmul_full_A_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_full_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[k][r]*B[k][c];
    }
}

static void fp_matmul_B_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[r][k]*B[c][k];
    }
}

static void fp_matmul_short_B_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_elem_t C[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C[r][c] += A[r][k]*B[c][k];
    }
}

static void fp_matmul_full_B_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_full_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[r][k]*B[c][k];
    }
}

static void fp_matmul_AB_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[k][r]*B[c][k];
    }
}

static void fp_matmul_short_AB_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_elem_t D[FP_DIM][FP_DIM], fp_elem_t C[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C[r][c] += A[k][r]*B[c][k];
    }
}

static void fp_matmul_full_AB_transposed(fp_elem_t A[FP_DIM][FP_DIM], fp_elem_t B[FP_DIM][FP_DIM], fp_full_t D[FP_DIM][FP_DIM], fp_full_t C_full[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < FP_DIM; k++)
        C_full[r][c] += A[k][r]*B[c][k];
    }
}

static void fp_matadd(fp_full_t sum[FP_DIM][FP_DIM], fp_full_t m1[FP_DIM][FP_DIM], fp_full_t m2[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++)
      sum[r][c] = m1[r][c] + m2[r][c];
}

// THIS IS A ROUNDING SHIFT! It also performs a saturating cast
static void fp_matshift(fp_full_t full[FP_DIM][FP_DIM], fp_elem_t out[FP_DIM][FP_DIM], int shift) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      // Bitshift and round element
      fp_full_t shifted = FP_ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
#ifndef FP_ELEM_T_IS_FLOAT
      fp_full_t elem = shifted > fp_elem_t_max ? fp_elem_t_max : (shifted < fp_elem_t_min ? fp_elem_t_min : shifted);
      out[r][c] = elem;
#else
      out[r][c] = shifted; // TODO should we also saturate when using floats?
#endif
    }
}

static void fp_matscale(fp_full_t full[FP_DIM][FP_DIM], fp_elem_t out[FP_DIM][FP_DIM], fp_acc_scale_t scale) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++) {
      // Bitshift and round element
      fp_full_t scaled = FP_ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef FP_ELEM_T_IS_FLOAT
      fp_full_t elem = scaled > fp_elem_t_max ? fp_elem_t_max : (scaled < fp_elem_t_min ? fp_elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

static void fp_matrelu(fp_elem_t in[FP_DIM][FP_DIM], fp_elem_t out[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++)
      out[r][c] = in[r][c] > 0 ? in[r][c] : 0;
}

static void fp_transpose(fp_elem_t in[FP_DIM][FP_DIM], fp_elem_t out[FP_DIM][FP_DIM]) {
  for (size_t r = 0; r < FP_DIM; r++)
    for (size_t c = 0; c < FP_DIM; c++)
      out[c][r] = in[r][c];
}

int fp_rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}


#ifdef FP_ELEM_T_IS_FLOAT
double fp_rand_double() {
    double a = (double)(fp_rand() % 128) / (double)(1 + (fp_rand() % 64));
    double b = (double)(fp_rand() % 128) / (double)(1 + (fp_rand() % 64));
    return a - b;
}
#endif

static void fp_printMatrix(fp_elem_t m[FP_DIM][FP_DIM]) {
  for (size_t i = 0; i < FP_DIM; ++i) {
    for (size_t j = 0; j < FP_DIM; ++j)
#ifndef FP_ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", fp_elem_t_to_elem_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

static void fp_printMatrixAcc(fp_acc_t m[FP_DIM][FP_DIM]) {
  for (size_t i = 0; i < FP_DIM; ++i) {
    for (size_t j = 0; j < FP_DIM; ++j)
#ifndef FP_ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", fp_acc_t_to_acc_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

static int fp_is_equal(fp_elem_t x[FP_DIM][FP_DIM], fp_elem_t y[FP_DIM][FP_DIM]) {
  for (size_t i = 0; i < FP_DIM; ++i)
    for (size_t j = 0; j < FP_DIM; ++j) {
#ifndef FP_ELEM_T_IS_FLOAT
      if (x[i][j] != y[i][j])
#else
      bool isnanx = fp_elem_t_isnan(x[i][j]);
      bool isnany = fp_elem_t_isnan(y[i][j]);

      if (x[i][j] != y[i][j] && !(isnanx && isnany))
#endif
          return 0;
    }
  return 1;
}

static int fp_is_equal_transposed(fp_elem_t x[FP_DIM][FP_DIM], fp_elem_t y[FP_DIM][FP_DIM]) {
  for (size_t i = 0; i < FP_DIM; ++i)
    for (size_t j = 0; j < FP_DIM; ++j) {
#ifndef FP_ELEM_T_IS_FLOAT
      if (x[i][j] != y[j][i])
#else
      bool isnanx = fp_elem_t_isnan(x[i][j]);
      bool isnany = fp_elem_t_isnan(y[j][i]);

      if (x[i][j] != y[j][i] && !(isnanx && isnany))
#endif
          return 0;
    }
  return 1;
}

// This is a GNU extension known as statment expressions
#define MAT_IS_EQUAL(dim_i, dim_j, x, y) \
    ({int result = 1; \
      for (size_t i = 0; i < dim_i; i++) \
        for (size_t j = 0; j < dim_j; ++j) { \
          if (x[i][j] != y[i][j]) { \
            result = 0; \
            break; \
          } \
        } \
      result;})

static uint64_t fp_read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}

#undef abs

#endif  // SRC_MAIN_C_GEMMINI_TESTUTILS_H
