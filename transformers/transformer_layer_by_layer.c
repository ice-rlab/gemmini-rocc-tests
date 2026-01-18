#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

static void counter_configure_extra(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0xff) << 4 | 0x8 | (counter_code & 0xff) << 12 | non_incremental << 31; //change this back to 64 later
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

static uint32_t counter_read_extra(size_t index) {
  uint32_t config_reg = (index & 0xff) << 4; //change this back to 64 later
  uint32_t res;
  gemmini_counter_access(res, config_reg);
  return res;
}

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
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

	uint64_t start, end;

	unsigned int total_macs_start, total_macs_end;
unsigned int bytes_loaded_a_start, bytes_loaded_a_end;
unsigned int bytes_loaded_b_start, bytes_loaded_b_end;
unsigned int bytes_loaded_d_start, bytes_loaded_d_end;
unsigned int bytes_read_start, bytes_read_end;
unsigned int rdma_bytes_rec_start, rdma_bytes_rec_end;
unsigned int wdma_bytes_sent_start, wdma_bytes_sent_end;

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

	total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ qkv_b, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("qkv %d cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", i, end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

		total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("softmax(Q * K) %d cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", head, end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("attn * V %d cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", head, end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

    }

    gemmini_fence();

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    // out_buf_acc = out_buf * Wo
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

    gemmini_fence();

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out_buf_acc = out_buf * Wo cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out = LN(out_buf_acc) cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();

end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("input = out + input cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{

	uint64_t start, end;

	unsigned int total_macs_start, total_macs_end;
unsigned int bytes_loaded_a_start, bytes_loaded_a_end;
unsigned int bytes_loaded_b_start, bytes_loaded_b_end;
unsigned int bytes_loaded_d_start, bytes_loaded_d_end;
unsigned int bytes_read_start, bytes_read_end;
unsigned int rdma_bytes_rec_start, rdma_bytes_rec_end;
unsigned int wdma_bytes_sent_start, wdma_bytes_sent_end;

    // out = FF1(input)
    // out = GELU(out)

	total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

	end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out = GELU(FF1(input)) cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

    // out_buf_acc = FF2(out)

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();


    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

	end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out = GELU(FF1(input)) cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

    // out = LN(out_buf_acc)

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

	end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out = GELU(FF1(input)) cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);

total_macs_start = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_start = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_start = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_start = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_start = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_start = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_start = counter_read_extra(WDMA_BYTES_SENT - 1);
start = read_cycles();

    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();

	end = read_cycles();
        total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

printf("out = GELU(FF1(input)) cycles: %llu, total_macs = %u, bytes_loaded_a = %u, bytes_loaded_b = %u, bytes_loaded_d = %u, bytes_read = %u, rdma_bytes_rec = %u, wdma_bytes_sent = %u\n", end-start, total_macs_end - total_macs_start, bytes_loaded_a_end - bytes_loaded_a_start, bytes_loaded_b_end - bytes_loaded_b_start, bytes_loaded_d_end - bytes_loaded_d_start, bytes_read_end - bytes_read_start, rdma_bytes_rec_end - rdma_bytes_rec_start, wdma_bytes_sent_end - wdma_bytes_sent_start);
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

    uint64_t start = read_cycles();

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

    uint64_t end = read_cycles();

    return end - start;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static const acc_t Wqkvo_b[4][hidden_dim]; \
    static const acc_t Wqkvo_cross_b[4][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[expansion_dim]; \
    static const acc_t ff2_b[hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static acc_t out_buf_acc[seq_len][hidden_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
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
    static const elem_t input[seq_len][hidden_dim]; \
    static const elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
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

unsigned int total_macs_end;
unsigned int bytes_loaded_a_end;
unsigned int bytes_loaded_b_end;
unsigned int bytes_loaded_d_end;
unsigned int bytes_read_end;
unsigned int rdma_bytes_rec_end;
unsigned int wdma_bytes_sent_end;

    gemmini_flush(0);

	counter_configure_extra(TOTAL_MACS - 1, TOTAL_MACS);
counter_configure_extra(BYTES_LOADED_A - 1, BYTES_LOADED_A);
counter_configure_extra(BYTES_LOADED_B - 1, BYTES_LOADED_B);
counter_configure_extra(BYTES_LOADED_D - 1, BYTES_LOADED_D);
counter_configure_extra(BYTES_READ - 1, BYTES_READ);
counter_configure_extra(RDMA_BYTES_REC - 1, RDMA_BYTES_REC);
counter_configure_extra(WDMA_BYTES_SENT - 1, WDMA_BYTES_SENT);

    PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/128, /*compression_factor=*/1);

	total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

    printf("FINAL TOTAL_MACS: %u\n", total_macs_end);
    printf("FINAL BYTES_LOADED_A: %u\n", bytes_loaded_a_end);
    printf("FINAL BYTES_LOADED_B: %u\n", bytes_loaded_b_end);
    printf("FINAL BYTES_LOADED_D: %u\n", bytes_loaded_d_end);
    printf("FINAL BYTES_READ: %u\n", bytes_read_end);
    printf("FINAL RDMA_BYTES_REC: %u\n", rdma_bytes_rec_end);
    printf("FINAL WDMA_BYTES_SENT: %u\n", wdma_bytes_sent_end);


    PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128, /*compression_factor=*/1);

	total_macs_end = counter_read_extra(TOTAL_MACS - 1);
bytes_loaded_a_end = counter_read_extra(BYTES_LOADED_A - 1);
bytes_loaded_b_end = counter_read_extra(BYTES_LOADED_B - 1);
bytes_loaded_d_end = counter_read_extra(BYTES_LOADED_D - 1);
bytes_read_end = counter_read_extra(BYTES_READ - 1);
rdma_bytes_rec_end = counter_read_extra(RDMA_BYTES_REC - 1);
wdma_bytes_sent_end = counter_read_extra(WDMA_BYTES_SENT - 1);

    printf("FINAL TOTAL_MACS: %u\n", total_macs_end);
    printf("FINAL BYTES_LOADED_A: %u\n", bytes_loaded_a_end);
    printf("FINAL BYTES_LOADED_B: %u\n", bytes_loaded_b_end);
    printf("FINAL BYTES_LOADED_D: %u\n", bytes_loaded_d_end);
    printf("FINAL BYTES_READ: %u\n", bytes_read_end);
    printf("FINAL RDMA_BYTES_REC: %u\n", rdma_bytes_rec_end);
    printf("FINAL WDMA_BYTES_SENT: %u\n", wdma_bytes_sent_end);


    exit(0);
}

