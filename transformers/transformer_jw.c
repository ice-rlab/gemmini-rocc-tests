#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include <math.h>
#include <stdlib.h>

#define MAX_SEQ_LEN     128
#define MAX_HIDDEN_DIM  768
#define MAX_EXP_DIM     3072

static elem_t A_q_buf[MAX_SEQ_LEN * MAX_EXP_DIM];
static elem_t B_q_buf[MAX_EXP_DIM * MAX_EXP_DIM];

// ===== Baseline buffers for SQNR measurement =====
static float ff1_lin_ref_buf [MAX_SEQ_LEN * MAX_EXP_DIM];   // FF1 linear (pre-GELU)
static float ff1_gelu_ref_buf[MAX_SEQ_LEN * MAX_EXP_DIM];   // FF1 after GELU
static float ff2_ref_buf     [MAX_SEQ_LEN * MAX_HIDDEN_DIM]; // FF2 output

// ================== local math helpers (avoid missing libm symbols) ==================

static inline float my_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;

    float g = x;
    for (int i = 0; i < 6; i++) {
        g = 0.5f * (g + x / g);
    }
    return g;
}

static inline float my_log10f(float x) {
    if (x <= 0.0f) {
        return -100.0f;
    }

    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get exponent and mantissa
    int exp = ((v.u >> 23) & 0xFF) - 127;             // real exponent
    v.u = (v.u & 0x7FFFFFu) | (127u << 23);           // mantissa in [1,2)
    float m = v.f;

    // approximation for log2(m)，m ∈ [1,2)
    float y = m - 1.0f;
    float log2_m = y * (1.346555f + y * (-0.356675f));

    float log2_x = (float)exp + log2_m;

    // log10(x) = log2(x) * log10(2)
    const float LOG10_2 = 0.30102999566f;
    return log2_x * LOG10_2;
}

static inline float my_tanhf(float x) {
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// -------- helper: float GELU ----------
static inline float gelu_float(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + my_tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// -------- helper: CPU matmul baseline -----------
static void matmul_cpu_ref(
    int M, int N, int K,
    const elem_t *A, int ldA,    // A: [M x K]
    const elem_t *B, int ldB,    // B: [K x N]
    const acc_t *bias,           // may be NULL; if not NULL, length N (repeating_bias = true)
    float *C, int ldC)           // C: [M x N] in float
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            if (bias) {
                sum += (float)bias[j];
            }

            for (int k = 0; k < K; k++) {
                float a = (float)A[i * ldA + k];
                float b = (float)B[k * ldB + j];
                sum += a * b;
            }

            C[i * ldC + j] = sum;
        }
    }
}

// -------- helper: SQNR + error rate ----------
static void measure_sqnr_and_error(
    const float *ref,      int ld_ref,
    const float *quant,    int ld_quant,
    int M, int N,
    const char *tag)
{
    double signal_pow = 0.0;
    double noise_pow  = 0.0;
    int total = M * N;
    int err_cnt = 0;
    const float thr = 1.0f;   // |error| > 1 计为 1 个 error

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float x = ref  [i * ld_ref   + j];
            float y = quant[i * ld_quant + j];
            float diff = x - y;

            signal_pow += (double)x * (double)x;
            noise_pow  += (double)diff * (double)diff;

            if (fabsf(diff) > thr)
                err_cnt++;
        }
    }

    float sqnr_db;
    if (noise_pow <= 0.0 || signal_pow <= 0.0) {
        sqnr_db = 100.0f;   // basically perfect
    } else {
        sqnr_db = 10.0f * my_log10f((float)(signal_pow / noise_pow));
    }

    float err_rate = (float)err_cnt / (float)total;

    printf("[SQNR] %s: SQNR = %.2f dB, err_rate(|err|>%.1f) = %.4f\n",
           tag, sqnr_db, thr, err_rate);
}


// ============================ perf counters ============================
typedef struct { uint64_t cycles; uint64_t calls; } perf_stat_t;

#define PERF_INIT(s) do { (s).cycles = 0; (s).calls = 0; } while(0)
#define PERF_ADD(ps, c) do { (ps)->cycles += (c); (ps)->calls++; } while(0)
#define PERF_BLOCK(pstat, CODE) do {      \
    uint64_t __t0 = read_cycles();          \
    { CODE }                                \
    uint64_t __t1 = read_cycles();          \
    PERF_ADD((pstat), (__t1 - __t0));       \
} while(0)

typedef struct {
  perf_stat_t q_matmul, k_matmul, v_matmul;   // Q/K/V linear layer
  perf_stat_t qkT_softmax;                    // QK^T (+ softmax activation)
  perf_stat_t attn_v;                         // Attn * V
  perf_stat_t wo_proj;                        // output projection Wo
  perf_stat_t ln1, resadd1;                   // LN + residual（after attention block）
  perf_stat_t ff1_gelu;                       // FFN1 (+ GELU)
  perf_stat_t ff2;                            // FFN2
  perf_stat_t ln2, resadd2;                   // LN + resudual（after FFN）
  perf_stat_t total_layer;                    // entire encoder/decoder layer
} layer_perf_t;


static uint32_t rng_state = 1234567;

static inline uint32_t fast_rand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

// [0, max) randint
static inline int rand_int(int max) {
    return (int)(fast_rand() % (uint32_t)max);
}

static inline void quantize_to_int8_copy(
    const elem_t* src,
    int ld_src,          
    elem_t* dst,
    int ld_dst,          
    int rows, int cols)  
{
    float max_abs = 0.0f;

    // 1. find max
    for (int r = 0; r < rows; r++) {
        const elem_t* srow = src + r * ld_src;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(srow[c]);
            if (a > max_abs) max_abs = a;
        }
    }

    // 2. compute scale，map max_abs -> 127
    float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;

    // 3. quantize to int8 
    for (int r = 0; r < rows; r++) {
        const elem_t* srow = src + r * ld_src;
        elem_t* drow = dst + r * ld_dst;
        for (int c = 0; c < cols; c++) {
            float v = srow[c] * scale;

            if (v > 127.0f) v = 127.0f;
            else if (v < -127.0f) v = -127.0f;

            int q = (int) lrintf(v);
            drow[c] = (elem_t) q;
        }
    }
}


void sample_submatrix(
    const elem_t *matrix,
    int rows, int cols,
    int sub_rows, int sub_cols,
    elem_t *submatrix,
    int *out_start_row,
    int *out_start_col
) {
    // boundary-check
    if (sub_rows > rows) sub_rows = rows;
    if (sub_cols > cols) sub_cols = cols;

    int max_start_row = rows - sub_rows;
    int max_start_col = cols - sub_cols;

    int start_row = (max_start_row > 0) ? rand_int(max_start_row + 1) : 0;
    int start_col = (max_start_col > 0) ? rand_int(max_start_col + 1) : 0;

    if (out_start_row) *out_start_row = start_row;
    if (out_start_col) *out_start_col = start_col;

    // copy sub matrix by rows
    for (int r = 0; r < sub_rows; r++) {
        const elem_t *src_row = matrix + (start_row + r) * cols + start_col;
        elem_t *dst_row = submatrix + r * sub_cols;
        memcpy(dst_row, src_row, sizeof(elem_t) * sub_cols);
    }
}

static layer_perf_t g_perf;
static inline void perf_reset(layer_perf_t *p) {
  PERF_INIT(p->q_matmul); PERF_INIT(p->k_matmul); PERF_INIT(p->v_matmul);
  PERF_INIT(p->qkT_softmax); PERF_INIT(p->attn_v); PERF_INIT(p->wo_proj);
  PERF_INIT(p->ln1); PERF_INIT(p->resadd1);
  PERF_INIT(p->ff1_gelu); PERF_INIT(p->ff2);
  PERF_INIT(p->ln2); PERF_INIT(p->resadd2);
  PERF_INIT(p->total_layer);
}

static inline void perf_report(const layer_perf_t *p, const char* tag) {
  uint64_t total = p->total_layer.cycles;
  if (total == 0) total = 1;
  #define PCT(x) (int)((100 * (x).cycles) / total)
  printf("\n[Perf] %s breakdown (cycles / %% / calls)\n", tag);
  printf("  Q    : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->q_matmul.cycles,  PCT(p->q_matmul),  (unsigned long long)p->q_matmul.calls);
  printf("  K    : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->k_matmul.cycles,  PCT(p->k_matmul),  (unsigned long long)p->k_matmul.calls);
  printf("  V    : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->v_matmul.cycles,  PCT(p->v_matmul),  (unsigned long long)p->v_matmul.calls);
  printf("  QK^T : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->qkT_softmax.cycles, PCT(p->qkT_softmax), (unsigned long long)p->qkT_softmax.calls);
  printf("  AttnV: %10llu  %3d%%  (%llu)\n", (unsigned long long)p->attn_v.cycles,      PCT(p->attn_v),      (unsigned long long)p->attn_v.calls);
  printf("  Wo   : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->wo_proj.cycles,     PCT(p->wo_proj),     (unsigned long long)p->wo_proj.calls);
  printf("  LN1  : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->ln1.cycles,         PCT(p->ln1),         (unsigned long long)p->ln1.calls);
  printf("  Res1 : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->resadd1.cycles,      PCT(p->resadd1),      (unsigned long long)p->resadd1.calls);
  printf("  FF1  : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->ff1_gelu.cycles,     PCT(p->ff1_gelu),     (unsigned long long)p->ff1_gelu.calls);
  printf("  FF2  : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->ff2.cycles,          PCT(p->ff2),          (unsigned long long)p->ff2.calls);
  printf("  LN2  : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->ln2.cycles,          PCT(p->ln2),          (unsigned long long)p->ln2.calls);
  printf("  Res2 : %10llu  %3d%%  (%llu)\n", (unsigned long long)p->resadd2.cycles,      PCT(p->resadd2),      (unsigned long long)p->resadd2.calls);
  printf("  TOTAL: %10llu  100%%   \n\n", (unsigned long long)total);
  #undef PCT
}

// ===========================================================================

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    const acc_t * qkv_bs[] = { Wq_b, Wk_b, Wv_b };

    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        elem_t * qkv_out = qkv_outs[i];

        perf_stat_t* statp =
            (i==0) ? &g_perf.q_matmul :
            (i==1) ? &g_perf.k_matmul :
                    &g_perf.v_matmul;

        PERF_BLOCK(
            statp,
            {
                const int M = seq_len;
                const int N = hidden_dim_compressed;
                const int K = hidden_dim;
        
                // A: [M x K], row-major, ld = hidden_dim
                quantize_to_int8_copy(
                    qkv_in,              /* ld_src  */ hidden_dim,
                    A_q_buf,             /* ld_dst  */ hidden_dim,
                    M, K);

                // B: [K x N], row-major, ld = hidden_dim
                quantize_to_int8_copy(
                    qkv_w,               /* ld_src  */ hidden_dim,
                    B_q_buf,             /* ld_dst  */ hidden_dim,
                    K, N);

                tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
                    /*A=*/ A_q_buf, /*B=*/ B_q_buf,
                    /*D=*/ qkv_b, /*C=*/ qkv_out,
                    /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
                    /*repeating_bias=*/ false,
                    false, /*transpose_B=*/ false,
                    false, false,
                    0,
                    WS);
            }
          );
    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        PERF_BLOCK(&g_perf.qkT_softmax, {
            const int M = seq_len;
            const int N = seq_len;
            const int K = hidden_dim_per_head;
        
            // A: [M x K], ld = hidden_dim
            quantize_to_int8_copy(
                A,          /* ld_src */ hidden_dim,
                A_q_buf,    /* ld_dst */ hidden_dim,
                M, K);
        
            quantize_to_int8_copy(
                B,          /* ld_src */ hidden_dim,
                B_q_buf,    /* ld_dst */ hidden_dim,
                K, N);

            tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
                /*A=*/ A_q_buf, /*B=*/ B_q_buf,
                /*D=*/ NULL, /*C=*/ C,
                /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                SOFTMAX, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
                /*repeating_bias=*/ false,
                false, /*transpose_B=*/ true,
                false, false,
                0,
                WS);
          });
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        PERF_BLOCK(&g_perf.attn_v, {
            const int M = seq_len;
            const int N = hidden_dim_per_head;
            const int K = seq_len;
        
            // A: [M x K], ld = seq_len
            quantize_to_int8_copy(
                A,         /* ld_src */ seq_len,
                A_q_buf,   /* ld_dst */ seq_len,
                M, K);
        
            // B: [K x N], ld = hidden_dim
            quantize_to_int8_copy(
                B,         /* ld_src */ hidden_dim,
                B_q_buf,   /* ld_dst */ hidden_dim,
                K, N);

            tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
                /*A=*/ A_q_buf, /*B=*/ B_q_buf,
                /*D=*/ NULL, /*C=*/ C,
                /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
                /*repeating_bias=*/ false,
                false, /*transpose_B=*/ false,
                false, false,
                0,
                WS);
          });
    }

    gemmini_fence();

    // out_buf_acc = out_buf * Wo
    PERF_BLOCK(&g_perf.wo_proj, {
        const int M = seq_len;
        const int N = hidden_dim;
        const int K = hidden_dim_compressed;
    
        // A: [M x K], ld = hidden_dim
        quantize_to_int8_copy(
            out_buf,      /* ld_src */ hidden_dim,
            A_q_buf,      /* ld_dst */ hidden_dim,
            M, K);
    
        // B: [K x N], ld = hidden_dim
        quantize_to_int8_copy(
            Wo,           /* ld_src */ hidden_dim,
            B_q_buf,      /* ld_dst */ hidden_dim,
            K, N);

        tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
            /*A=*/ out_buf, /*B=*/ Wo,
            /*D=*/ Wo_b, /*C=*/ out_buf_acc,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            true, false,
            0,
            WS);
    });

    gemmini_fence();

    // out = LN(out_buf_acc)
    PERF_BLOCK(&g_perf.ln1, {
        tiled_norm_auto(seq_len, hidden_dim,
            (acc_t*)out_buf_acc, (elem_t*)out,
            ACC_SCALE_IDENTITY,
            LAYERNORM, WS);
      });

    // input = out + input
    PERF_BLOCK(&g_perf.resadd1, {
        tiled_resadd_auto(seq_len, hidden_dim,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            input,
            out,
            resadd_out,
            /*relu=*/ false,
            WS);
      });

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    const int M  = seq_len;
    const int N1 = expansion_dim;
    const int K1 = hidden_dim;
    const int N2 = hidden_dim;
    const int K2 = expansion_dim;

    // ================== FF1 baseline (no quant) ==================
    // FF1 linear: input [M x K1] * ff1_w [K1 x N1] + ff1_b
    matmul_cpu_ref(
        M, N1, K1,
        /*A=*/ input,        /*ldA=*/ hidden_dim,
        /*B=*/ ff1_w,        /*ldB=*/ expansion_dim,
        /*bias=*/ ff1_b,
        /*C=*/ ff1_lin_ref_buf, /*ldC=*/ expansion_dim);

    // GELU on baseline
    for (int i = 0; i < M; i++) {
        float *dst_row = ff1_gelu_ref_buf + i * N1;
        float *src_row = ff1_lin_ref_buf + i * N1;
        for (int j = 0; j < N1; j++) {
            dst_row[j] = gelu_float(src_row[j]);
        }
    }

    // out = FF1(input)
    // out = GELU(out)
    PERF_BLOCK(&g_perf.ff1_gelu, {    
        // A: [M x K], ld = hidden_dim
        quantize_to_int8_copy(
            input,        /* ld_src */ hidden_dim,
            A_q_buf,      /* ld_dst */ hidden_dim,
            M, K1);
    
        // B: [K x N], ld = expansion_dim
        quantize_to_int8_copy(
            ff1_w,        /* ld_src */ expansion_dim,
            B_q_buf,      /* ld_dst */ expansion_dim,
            K1, N1);

        tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
            /*A=*/ A_q_buf, /*B=*/ B_q_buf,
            /*D=*/ ff1_b, /*C=*/ out_buf,
            /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
            /*repeating_bias=*/ true,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
      });

    gemmini_fence();

    // Compare FF1 baseline (GELU) vs Gemmini result in out_buf
    measure_sqnr_and_error(
        /*ref   =*/ ff1_gelu_ref_buf,          /*ld_ref   =*/ expansion_dim,
        /*quant =*/ (const float*)out_buf,     /*ld_quant =*/ expansion_dim,
        M, N1,
        "FF1 (GELU output)");

        // ================== FF2 baseline (no quant) ==================
    // Baseline FF2 uses baseline GELU output as input
    matmul_cpu_ref(
        M, N2, K2,
        /*A=*/ (const elem_t*)ff1_gelu_ref_buf, /*ldA=*/ expansion_dim,
        /*B=*/ ff2_w,                           /*ldB=*/ hidden_dim,
        /*bias=*/ ff2_b,
        /*C=*/ ff2_ref_buf,                     /*ldC=*/ hidden_dim);

    for (int i = 0; i < M; i++) {
        elem_t *dst = out_buf + i * expansion_dim;
        float  *src = ff1_gelu_ref_buf + i * expansion_dim;
        for (int j = 0; j < N1; j++) {
            dst[j] = (elem_t)src[j];
        }
    }

    // out_buf_acc = FF2(out)
    PERF_BLOCK(&g_perf.ff2, {
        // A: [M x K], ld = expansion_dim
        quantize_to_int8_copy(
            out_buf,      /* ld_src */ expansion_dim,
            A_q_buf,      /* ld_dst */ expansion_dim,
            M, K2);
    
        // B: [K x N], ld = hidden_dim
        quantize_to_int8_copy(
            ff2_w,        /* ld_src */ hidden_dim,
            B_q_buf,      /* ld_dst */ hidden_dim,
            K2, N2);

        tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
            /*A=*/ A_q_buf, /*B=*/ B_q_buf,
            /*D=*/ ff2_b, /*C=*/ out_buf_acc,
            /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ true,
            false, /*transpose_B=*/ false,
            true, false,
            0,
            WS);
    });

    gemmini_fence();

    // Compare FF2 baseline vs Gemmini accumulator output
    measure_sqnr_and_error(
        /*ref   =*/ ff2_ref_buf,                 /*ld_ref   =*/ hidden_dim,
        /*quant =*/ (const float*)out_buf_acc,   /*ld_quant =*/ hidden_dim,
        M, N2,
        "FF2 (pre-LN output)");

    // out = LN(out_buf_acc)
    PERF_BLOCK(&g_perf.ln2, {
        tiled_norm_auto(seq_len, hidden_dim,
            (acc_t*)out_buf_acc, (elem_t*)out,
            ACC_SCALE_IDENTITY,
            LAYERNORM, WS);
      });

    gemmini_fence();

    // out = out + input
    PERF_BLOCK(&g_perf.resadd2, {
        tiled_resadd_auto(seq_len, hidden_dim,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            out,
            input,
            out,
            /*relu=*/ false,
            WS);
      });

    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b,
        const acc_t * Wo_cross_b,

        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    perf_reset(&g_perf);

    uint64_t start = read_cycles();
    PERF_INIT(g_perf.total_layer); 
    uint64_t t0 = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,

        Wq_b, Wk_b, Wv_b,
        Wo_b,

        Q_buf, K_buf, V_buf,
        attn_buf, out_buf, out_buf_acc);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,

            Wq_cross_b, Wk_cross_b, Wv_cross_b,
            Wo_cross_b,

            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf, out_buf_acc);

    uint64_t t1 = read_cycles();
    PERF_ADD(&g_perf.total_layer, (t1 - t0));

    perf_report(&g_perf, "Encoder/Decoder Layer");

    return t1 - start;
}

// yes
static inline int8_t seed8(int i){ return (int8_t)((1103515245u*i + 12345u)>>24) - 64; } // around[-64,63]
// void fill_input(elem_t* p, int n){ for(int i=0;i<n;i++) p[i] = (elem_t)(seed8(i)%13); }  
// void fill_w(elem_t* p, int n){ for(int i=0;i<n;i++) p[i] = (elem_t)(seed8(7*i+3)%7); }
// void fill_b(acc_t* p, int n){ for(int i=0;i<n;i++) p[i] = (acc_t)(seed8(5*i+1)%17); }
void fill_input(elem_t* p, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t x = 1103515245u * i + 12345u;
        float u = ((x >> 8) & 0xFFFF) / 65535.0f;  // [0,1]
        float centered = (u - 0.5f) * 3.0f;        // [-1.5, 1.5]
        p[i] = centered;
    }
}

// deterministic pseudo-random float in [0,1]
static inline float frand01(uint32_t seed) {
    uint32_t x = 1103515245u * seed + 12345u;
    return (x & 0xFFFFFF) / 16777215.0f;
}

// deterministic float in [-1,1]
static inline float frand(uint32_t seed) {
    return frand01(seed) * 2.0f - 1.0f;
}

void fill_w(elem_t* w, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    int n = fan_in * fan_out;

    for (int i = 0; i < n; i++) {
        w[i] = frand(i) * limit;      // Uniform(-limit, +limit)
    }
}

void fill_b(acc_t* b, int out_dim) {
    for (int i = 0; i < out_dim; i++)
        b[i] = 0.0f;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    \
    static elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static acc_t Wqkvo_b[4][hidden_dim]; \
    static acc_t Wqkvo_cross_b[4][hidden_dim]; \
    static elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static acc_t ff1_b[expansion_dim]; \
    static acc_t ff2_b[hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static acc_t out_buf_acc[seq_len][hidden_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    fill_w(&Wqkvo[0][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo[1][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo[2][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo[3][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo_cross[0][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo_cross[1][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo_cross[2][0][0], hidden_dim, hidden_dim); \
    fill_w(&Wqkvo_cross[3][0][0], hidden_dim, hidden_dim); \
    fill_b(&Wqkvo_b[0][0], hidden_dim); \
    fill_b(&Wqkvo_b[1][0], hidden_dim); \
    fill_b(&Wqkvo_b[2][0], hidden_dim); \
    fill_b(&Wqkvo_b[3][0], hidden_dim); \
    fill_b(&Wqkvo_cross_b[0][0], hidden_dim); \
    fill_b(&Wqkvo_cross_b[1][0], hidden_dim); \
    fill_b(&Wqkvo_cross_b[2][0], hidden_dim); \
    fill_b(&Wqkvo_cross_b[3][0], hidden_dim); \
    fill_w(&ff_w[0][0],          hidden_dim,   expansion_dim); \
    fill_w(&ff_w[1][0],          expansion_dim, hidden_dim); \
    fill_b(ff1_b,                  expansion_dim); \
    fill_b(ff2_b,                  hidden_dim); \
    \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            compression_factor, \
            \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            \
            Wqkvo_b[0], Wqkvo_b[1], Wqkvo_b[2], \
            Wqkvo_b[2], \
            Wqkvo_cross_b[0], Wqkvo_cross_b[1], Wqkvo_cross_b[2], \
            Wqkvo_cross_b[3], \
            \
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, out_buf_acc, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define PRINT_ENCODER_DECODER(name, is_encoder, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor) { \
    static elem_t input[seq_len][hidden_dim]; \
    static elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
    \
    fill_input(&input[0][0],  seq_len*hidden_dim);  \
    fill_input(&enc_out[0][0],seq_len*hidden_dim);  \
    \
    uint64_t cycles = ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, is_encoder ? NULL : enc_out, output); \
    \
    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n", \
            name, type_str, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor); \
    printf("%s cycles: %llu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/128, /*compression_factor=*/1);

    PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128, /*compression_factor=*/1);

    exit(0);
}

