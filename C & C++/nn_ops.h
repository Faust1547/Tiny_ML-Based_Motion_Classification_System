#ifndef NN_OPS_H
#define NN_OPS_H

#include <stdint.h>
#include "nn_weights.h"

// ------------------------------------------------------------
// Final combined NN operators
// 02: SRAM saving through immediate quantization and activation reuse
// 03: fixed-point preprocessing and optimized Conv1D
// 04: quantized smoothing and confidence-margin helper
// ------------------------------------------------------------

inline int8_t clamp_int8(int32_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}

inline int32_t round_div(int32_t x, int32_t d) {
  if (d <= 0) return 0;
  int32_t offset = d / 2;
  if (x >= 0) return (x + offset) / d;
  return -(((-x) + offset) / d);
}

// Fixed-point preprocessing: raw int16 IMU sample -> quantized int8 NN input.
// Uses PRE_S, PRE_Z, and PRE_SHIFT constants from nn_weights.h.
inline int8_t preprocess_one_channel_fixed(int16_t raw, int c) {
  int32_t acc = (int32_t)raw * PRE_S[c] + PRE_Z[c];

  if (PRE_SHIFT[c] > 0) {
    int32_t rounding = (int32_t)1 << (PRE_SHIFT[c] - 1);
    acc += (acc >= 0) ? rounding : -rounding;
  }

  int32_t q = acc >> PRE_SHIFT[c];
  return clamp_int8(q);
}

inline void preprocess_one_sample_fixed(
    int16_t ax, int16_t ay, int16_t az,
    int16_t gx, int16_t gy, int16_t gz,
    int8_t out[NN_INPUT_CH]
) {
  out[0] = preprocess_one_channel_fixed(ax, 0);
  out[1] = preprocess_one_channel_fixed(ay, 1);
  out[2] = preprocess_one_channel_fixed(az, 2);
  out[3] = preprocess_one_channel_fixed(gx, 3);
  out[4] = preprocess_one_channel_fixed(gy, 4);
  out[5] = preprocess_one_channel_fixed(gz, 5);
}

// Optional smoothing on already-quantized input. This keeps the SRAM-saving
// design because no int16 raw_samples[][] buffer is needed.
inline void smooth_quantized_samples_moving_average(int8_t data[NN_INPUT_LEN][NN_INPUT_CH]) {
  for (int c = 0; c < NN_INPUT_CH; c++) {
    int8_t prev = data[0][c];
    for (int i = 1; i < NN_INPUT_LEN - 1; i++) {
      int8_t cur = data[i][c];
      int8_t next = data[i + 1][c];
      int32_t sum = (int32_t)prev + (int32_t)cur + (int32_t)next;
      data[i][c] = clamp_int8(round_div(sum, 3));
      prev = cur;
    }
  }
}

inline int8_t requantize_int8(int32_t acc, int32_t mult, uint8_t shift) {
  int64_t scaled = (int64_t)acc * (int64_t)mult;

  if (shift > 0) {
    int64_t rounding = (int64_t)1 << (shift - 1);
    scaled += (scaled >= 0) ? rounding : -rounding;
  }

  int32_t q = (int32_t)(scaled >> shift);
  return clamp_int8(q);
}

inline void relu(int8_t *data, int len) {
  for (int i = 0; i < len; i++) {
    if (data[i] < 0) data[i] = 0;
  }
}

// Helper used for the two boundary samples of SAME-padding Conv1D.
inline void conv1d_boundary_sample(
    const int8_t *input, int8_t *output,
    const int8_t weights[], const int32_t bias[],
    int i, int len, int in_ch, int out_ch,
    int32_t rq_mult, uint8_t rq_shift
) {
  const int kernel = 3;
  for (int oc = 0; oc < out_ch; oc++) {
    int32_t sum = bias[oc];
    for (int k = 0; k < kernel; k++) {
      int idx = i + k - 1;
      if (idx < 0 || idx >= len) continue;

      const int8_t *w = &weights[(oc * kernel + k) * in_ch];
      const int8_t *x = &input[idx * in_ch];
      for (int ic = 0; ic < in_ch; ic++) {
        sum += (int32_t)w[ic] * (int32_t)x[ic];
      }
    }
    output[i * out_ch + oc] = requantize_int8(sum, rq_mult, rq_shift);
  }
}

// Optimized 3-tap Conv1D: boundary checks only at the two ends, no padding
// checks in the middle region.
inline void conv1d(
    const int8_t *input, int8_t *output,
    const int8_t weights[], const int32_t bias[],
    int len, int in_ch, int out_ch,
    int32_t rq_mult, uint8_t rq_shift
) {
  const int kernel = 3;
  if (len <= 0) return;

  conv1d_boundary_sample(input, output, weights, bias, 0, len, in_ch, out_ch, rq_mult, rq_shift);

  for (int i = 1; i < len - 1; i++) {
    const int8_t *x0 = &input[(i - 1) * in_ch];
    const int8_t *x1 = &input[i * in_ch];
    const int8_t *x2 = &input[(i + 1) * in_ch];

    for (int oc = 0; oc < out_ch; oc++) {
      int32_t sum = bias[oc];

      const int8_t *w0 = &weights[(oc * kernel + 0) * in_ch];
      const int8_t *w1 = &weights[(oc * kernel + 1) * in_ch];
      const int8_t *w2 = &weights[(oc * kernel + 2) * in_ch];

      for (int ic = 0; ic < in_ch; ic++) {
        sum += (int32_t)w0[ic] * (int32_t)x0[ic];
        sum += (int32_t)w1[ic] * (int32_t)x1[ic];
        sum += (int32_t)w2[ic] * (int32_t)x2[ic];
      }

      output[i * out_ch + oc] = requantize_int8(sum, rq_mult, rq_shift);
    }
  }

  if (len > 1) {
    conv1d_boundary_sample(input, output, weights, bias, len - 1, len, in_ch, out_ch, rq_mult, rq_shift);
  }
}

// In-place maxpool for activation-buffer reuse. It removes the separate
// pool1_out[][] SRAM buffer used by the baseline.
inline void maxpool1d_inplace_conv1(int8_t buffer[][NN_CONV1_OUT_CH], int len) {
  int out_len = len / 2;
  for (int i = 0; i < out_len; i++) {
    for (int c = 0; c < NN_CONV1_OUT_CH; c++) {
      int8_t a = buffer[i * 2 + 0][c];
      int8_t b = buffer[i * 2 + 1][c];
      buffer[i][c] = (a > b) ? a : b;
    }
  }
}

inline void global_avg_pool(const int8_t input[][NN_CONV2_OUT_CH], int8_t output[NN_CONV2_OUT_CH], int len) {
  for (int c = 0; c < NN_CONV2_OUT_CH; c++) {
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
      sum += (int32_t)input[i][c];
    }
    int32_t avg = round_div(sum, len);
    output[c] = requantize_int8(avg, RQ_MULT_GAP, RQ_SHIFT_GAP);
  }
}

inline void dense(const int8_t input[], int8_t output[], const int8_t weights[], const int32_t bias[], int in_dim, int out_dim, int32_t rq_mult, uint8_t rq_shift) {
  for (int o = 0; o < out_dim; o++) {
    int32_t sum = bias[o];
    int base = o * in_dim;
    for (int i = 0; i < in_dim; i++) {
      sum += (int32_t)weights[base + i] * (int32_t)input[i];
    }
    output[o] = requantize_int8(sum, rq_mult, rq_shift);
  }
}

inline int argmax(const int8_t data[], int len) {
  int best = 0;
  int8_t best_val = data[0];
  for (int i = 1; i < len; i++) {
    if (data[i] > best_val) {
      best_val = data[i];
      best = i;
    }
  }
  return best;
}

inline int16_t prediction_margin_int16(const int8_t *logits, int n) {
  int16_t best = -32768;
  int16_t second = -32768;

  for (int i = 0; i < n; i++) {
    int16_t v = logits[i];

    if (v > best) {
      second = best;
      best = v;
    } else if (v > second) {
      second = v;
    }
  }

  return best - second;
}

#endif // NN_OPS_H
