/*
 * Copyright (C) 2023 Ayan Shafqat <ayan.x.shafqat@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef A5EQ_DSP_BIQUAD_HPP_
#define A5EQ_DSP_BIQUAD_HPP_

#include <algorithm>
#include <cstddef>

#define ARM_NEON_OPTIMIZATION (3)
#define AVX2_FMA_OPTIMIZATION (2)
#define SSE2_OPTIMIZATION (1)
#define NO_OPTIMIZATION (0)

#if (defined(__aarch64__)) && (defined(__ARM_NEON) || defined(__ARM_NEON__))

// Include NEON intrinsic header file
#include <arm_neon.h>
#define DSP_OPTIMIZATION (ARM_NEON_OPTIMIZATION)
#pragma message "Compiling with ARM NEON optimizations"

#elif defined(__x86_64__) && defined(__FMA__)

// Include FMA intrinsic header file
#include <immintrin.h>
#define DSP_OPTIMIZATION (AVX2_FMA_OPTIMIZATION)
#pragma message "Compiling with AVX2 FMA optimizations"

#elif defined(__x86_64__) && defined(__SSE2__)

// Include SSE+ intrinsic header file
#include <xmmintrin.h>
#define DSP_OPTIMIZATION (SSE2_OPTIMIZATION)
#pragma message "Compiling with SSE2 optimizations"

#else

#define DSP_OPTIMIZATION (NO_OPTIMIZATION)
#pragma message "Compiling with no specific optimizations"

#endif // Select optimization levels

/************************
 * Filter implementations
 ************************/

#if (DSP_OPTIMIZATION == ARM_NEON_OPTIMIZATION)

// The PROCESS_BIQUAD macro would be redefined for NEON
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                     \
  do {                                                                         \
    /* y0 = (b0 * x0) + s1 */                                                  \
    y0 = vmlaq_f64(s1, b0, x0);                                                \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                      \
    s1 = vmlaq_f64(s2, b1, x0);                                                \
    s1 = vmlaq_f64(s1, a1, y0);                                                \
    /* s2 = (b2 * x0) + (a2 * y0) */                                           \
    s2 = vmlaq_f64(vmulq_f64(a2, y0), b2, x0);                                 \
  } while (0)

/**
 * @brief Process biquad filter for a single stage of biquad
 *
 * @details
 *
 * Inspired from the author's old code with 4 channel biquad here:
 *
 * https://gist.github.com/ashafq/0db953125a033b783c6e100acd5e64d9
 *
 * This is optimized for AARCH64 NEON
 *
 * @param[in] coeff Filter coefficients
 * @param[in,out] state Filter state
 * @param[out] dst Destination/output buffer
 * @param[in] src Source/input buffer
 * @param len Number of samples to process
 *
 * @note Filter is executed in Direct form II topology
 * @see https://ccrma.stanford.edu/~jos/fp/Transposed_Direct_Forms.html
 *
 * @note Memory segments in @p src and @p dst may overlap
 */
void process_biquad_stereo(const double *coeff, double *state, float *out_left,
                           const float *in_left, float *out_right,
                           const float *in_right, size_t frames) {
  // Copy the same coefficients for both left and right channels
  float64x2_t b0 = vdupq_n_f64(coeff[0]);
  float64x2_t b1 = vdupq_n_f64(coeff[1]);
  float64x2_t b2 = vdupq_n_f64(coeff[2]);
  float64x2_t a1 = vdupq_n_f64(-coeff[3]);
  float64x2_t a2 = vdupq_n_f64(-coeff[4]);

  float64x2_t s1 = vld1q_f64(state + 0);
  float64x2_t s2 = vld1q_f64(state + 2);

  size_t simd_frames = frames & (~static_cast<size_t>(3));
  size_t count = 0;

  while (count < simd_frames) {
    // Load input samples into SSE registers and convert to double
    float32x4_t x_left = vld1q_f32(in_left + count);   // {l0, l1, l2, l3}
    float32x4_t x_right = vld1q_f32(in_right + count); // {l0, l1, l2, l3}

    float64x2_t x_left_0 =
        vcvt_f64_f32(vget_low_f32(x_left)); // {l0, l1} (double)
    float64x2_t x_right_0 =
        vcvt_f64_f32(vget_low_f32(x_right)); // {r0, r1} (double)

    float64x2_t x_left_1 =
        vcvt_f64_f32(vget_high_f32(x_left)); // {l2, l3} (double)
    float64x2_t x_right_1 =
        vcvt_f64_f32(vget_high_f32(x_right)); // {r2, r3} (double)

    // Interleave left and right inputs for stereo pair
    float64x2_t x0 = vtrn1q_f64(x_left_0, x_right_0); // {l0, r0}
    float64x2_t x1 = vtrn2q_f64(x_left_0, x_right_0); // {l1, r1}
    float64x2_t x2 = vtrn1q_f64(x_left_1, x_right_1); // {l2, r2}
    float64x2_t x3 = vtrn2q_f64(x_left_1, x_right_1); // {l3, r3}

    // Outputs
    float64x2_t y0, y1, y2, y3;

    PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x1, y1, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x2, y2, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x3, y3, b0, b1, b2, a1, a2, s1, s2);

    float64x2_t y_left_0 = vuzp1q_f64(y0, y1);  // {l0, l1}
    float64x2_t y_right_0 = vuzp2q_f64(y0, y1); // {r0, r1}
    float64x2_t y_left_1 = vuzp1q_f64(y2, y3);  // {l2, l3}
    float64x2_t y_right_1 = vuzp2q_f64(y2, y3); // {r2, r3}

    // Convert the output to float and combine them into two registers
    float32x4_t y_left = vcombine_f32(
        vcvt_f32_f64(y_left_0), vcvt_f32_f64(y_left_1)); // {l0, l1, l2, l3}
    float32x4_t y_right = vcombine_f32(
        vcvt_f32_f64(y_right_0), vcvt_f32_f64(y_right_1)); // {r0, r1, r2, r3}

    // Store the results
    vst1q_f32(out_left + count, y_left);
    vst1q_f32(out_right + count, y_right);

    count += 4;
  }

  // Process any remaining frames
  while (count < frames) {
    // Load single float values and create float32x2_t vectors
    float64x1_t x_left = vdup_n_f64(static_cast<double>(in_left[count]));
    float64x1_t x_right = vdup_n_f64(static_cast<double>(in_right[count]));

    // Interleave left and right inputs for stereo pair
    float64x2_t x0 = vcombine_f64(x_left, x_right);

    // Outputs
    float64x2_t y0;

    PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);

    // y0 = {yl, yr}
    float64x1_t yl = vget_low_f64(y0);  // {yl}
    float64x1_t yr = vget_high_f64(y0); // {yr}

    // Store the results
    out_left[count] = static_cast<float32_t>(vget_lane_f64(yl, 0));
    out_right[count] = static_cast<float32_t>(vget_lane_f64(yr, 0));

    ++count;
  }

  // Store states
  vst1q_f64(state + 0, s1);
  vst1q_f64(state + 2, s2);
}

#undef PROCESS_BIQUAD // No longer needed

#endif // ARM_NEON_OPTIMIZATION

#if (DSP_OPTIMIZATION == AVX2_FMA_OPTIMIZATION) ||                             \
    (DSP_OPTIMIZATION == SSE2_OPTIMIZATION)

/**
 * Since FMA and SSE2 are mostly shared code, it's best to consolidate them
 */
#if (DSP_OPTIMIZATION == AVX2_FMA_OPTIMIZATION)
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                     \
  do {                                                                         \
    /* y0 = (b0 * x0) + s1 */                                                  \
    y0 = _mm_fmadd_pd(b0, x0, s1);                                             \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                      \
    s1 = _mm_fmadd_pd(b1, x0, _mm_fmadd_pd(a1, y0, s2));                       \
    /* s2 = (b2 * x0) + (a2 * y0) */                                           \
    s2 = _mm_fmadd_pd(b2, x0, _mm_mul_pd(a2, y0));                             \
  } while (false)
#elif (DSP_OPTIMIZATION == SSE2_OPTIMIZATION)
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                     \
  do {                                                                         \
    /* y0 = (b0 * x0) + s1 */                                                  \
    y0 = _mm_add_pd(_mm_mul_pd(b0, x0), s1);                                   \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                      \
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x0), _mm_mul_pd(a1, y0)), s2);   \
    /* s2 = (b2 * x0) + (a2 * y0) */                                           \
    s2 = _mm_add_pd(_mm_mul_pd(b2, x0), _mm_mul_pd(a2, y0));                   \
  } while (false)
#endif // DSP_OPTIMIZATION

/**
 * @brief Process biquad filter for a single stage of biquad
 *
 * @details
 *
 * Inspired from the author's old code with 4 channel biquad here:
 *
 * https://gist.github.com/ashafq/0db953125a033b783c6e100acd5e64d9
 *
 * @param[in] coeff Filter coefficients
 * @param[in,out] state Filter state
 * @param[out] dst Destination/output buffer
 * @param[in] src Source/input buffer
 * @param len Number of samples to process
 *
 * @note Filter is executed in Direct form II topology
 * @see https://ccrma.stanford.edu/~jos/fp/Transposed_Direct_Forms.html
 *
 * @note Memory segments in @p src and @p dst may overlap
 */
static void process_biquad_stereo(const double *coeff, double *state,
                                  float *out_left, const float *in_left,
                                  float *out_right, const float *in_right,
                                  size_t frames) {
  // Copy the same coefficients for both left and right channels
  __m128d b0 = _mm_set1_pd(coeff[0]);
  __m128d b1 = _mm_set1_pd(coeff[1]);
  __m128d b2 = _mm_set1_pd(coeff[2]);
  __m128d a1 = _mm_set1_pd(-coeff[3]);
  __m128d a2 = _mm_set1_pd(-coeff[4]);

  __m128d s1 = _mm_loadu_pd(state + 0);
  __m128d s2 = _mm_loadu_pd(state + 2);

  size_t simd_frames = frames & (~static_cast<size_t>(3));
  size_t count = 0;

  for (count = 0; count < simd_frames; count += 4) {
    // Load input samples into SSE registers and convert to double
    __m128 x_left = _mm_loadu_ps(in_left + count);   // {l0, l1, l2, l3}
    __m128 x_right = _mm_loadu_ps(in_right + count); // {r0, r1, r2, r3}

    __m128d x_left_0 = _mm_cvtps_pd(x_left);   // {l0, l1} (double)
    __m128d x_right_0 = _mm_cvtps_pd(x_right); // {r0, r1} (double)
    __m128d x_left_1 =
        _mm_cvtps_pd(_mm_movehl_ps(x_left, x_left)); // {l2, l3} (double)
    __m128d x_right_1 =
        _mm_cvtps_pd(_mm_movehl_ps(x_right, x_right)); // {r2, r3} (double)

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left_0, x_right_0); // {l0, r0}
    __m128d x1 = _mm_unpackhi_pd(x_left_0, x_right_0); // {l1, r1}
    __m128d x2 = _mm_unpacklo_pd(x_left_1, x_right_1); // {l2, r2}
    __m128d x3 = _mm_unpackhi_pd(x_left_1, x_right_1); // {l3, r3})

    // Outputs
    __m128d y0, y1, y2, y3;

    PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x1, y1, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x2, y2, b0, b1, b2, a1, a2, s1, s2);
    PROCESS_BIQUAD(x3, y3, b0, b1, b2, a1, a2, s1, s2);

    // De-interleave the pairs
    __m128d y_left_0 = _mm_unpacklo_pd(y0, y1);  // {l0, l1}
    __m128d y_right_0 = _mm_unpackhi_pd(y0, y1); // {r0, r1}
    __m128d y_left_1 = _mm_unpacklo_pd(y2, y3);  // {l2, l3}
    __m128d y_right_1 = _mm_unpackhi_pd(y2, y3); // {r2, r3}

    // Convert the output to float and pack them into two registers
    __m128 y_left = _mm_movelh_ps(_mm_cvtpd_ps(y_left_0),
                                  _mm_cvtpd_ps(y_left_1)); // {l0, l1, l2, l3}

    __m128 y_right = _mm_movelh_ps(_mm_cvtpd_ps(y_right_0),
                                   _mm_cvtpd_ps(y_right_1)); // {r0, r1, r2, r3}

    _mm_storeu_ps(out_left + count, y_left);
    _mm_storeu_ps(out_right + count, y_right);
  }

  // Process any remaining frames
  while (count < frames) {
    // Load input samples into SSE registers and convert to double
    __m128d x_left = _mm_cvtps_pd(_mm_load_ss(in_left + count));
    __m128d x_right = _mm_cvtps_pd(_mm_load_ss(in_right + count));

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left, x_right);

    // Outputs
    __m128d y0;

    PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);

    // y0 = {yl, yr}
    __m128d yl = _mm_unpacklo_pd(y0, y0); // {yl, yl}
    __m128d yr = _mm_unpackhi_pd(y0, y0); // {yr, yr}

    // Store the result into out_left and out_right
    _mm_store_ss(out_left + count, _mm_cvtpd_ps(yl));
    _mm_store_ss(out_right + count, _mm_cvtpd_ps(yr));

    ++count;
  }

  // Store states
  _mm_storeu_pd(state + 0, s1);
  _mm_storeu_pd(state + 2, s2);
}

#undef PROCESS_BIQUAD // No longer needed

#endif // DSP_OPTIMIZATION == {AVX2_FMA_OPTIMIZATION, SSE2_OPTIMIZATION}

#if (DSP_OPTIMIZATION == NO_OPTIMIZATION)

/**
 * @brief Process biquad filter for a single stage of biquad
 *
 * @param[in] coeff Filter coefficients
 * @param[in,out] state Filter state
 * @param[out] dst Destination/output buffer
 * @param[in] src Source/input buffer
 * @param len Number of samples to process
 *
 * @note Filter is executed in Direct form II topology
 * @see https://ccrma.stanford.edu/~jos/fp/Transposed_Direct_Forms.html
 *
 * @note Memory segments in @p src and @p dst may overlap
 */
static void process_biquad_stereo(const double *coeff, double *state,
                                  float *out_left, const float *in_left,
                                  float *out_right, const float *in_right,
                                  size_t frames) {
  const auto b0 = *coeff++;
  const auto b1 = *coeff++;
  const auto b2 = *coeff++;
  const auto a1 = *coeff++;
  const auto a2 = *coeff++;

  // Process left channel
  auto s1 = state[0];
  auto s2 = state[2];

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_left[i]);

    auto out = (b0 * in) + s1;
    s1 = (b1 * in) - (a1 * out) + s2;
    s2 = (b2 * in) - (a2 * out);

    out_left[i] = static_cast<float>(out);
  }

  state[0] = s1;
  state[2] = s2;

  // Process right channel
  s1 = state[1];
  s2 = state[3];

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_right[i]);

    auto out = (b0 * in) + s1;
    s1 = (b1 * in) - (a1 * out) + s2;
    s2 = (b2 * in) - (a2 * out);

    out_right[i] = static_cast<float>(out);
  }

  state[1] = s1;
  state[3] = s2;
}
#endif // (DSP_OPTIMIZATION == NO_OPTIMIZATION)

/**
 * @brief Process biquad filter for a single stage of biquad and transition
 *
 * @param[in] old_coeff Old filter coefficients
 * @param[in] new_coeff New filter coefficients
 * @param[in,out] state Filter state
 * @param[out] dst Destination/output buffer
 * @param[in] src Source/input buffer
 * @param len Number of samples to process
 *
 * @note Memory segments in @p src and @p dst may overlap
 */
static void process_biquad_stereo_xfade(const double *old_coeff,
                                        const double *new_coeff, double *state,
                                        float *out_left, const float *in_left,
                                        float *out_right, const float *in_right,
                                        size_t frames) {
  // Implement the xfade utilizing the other biquad routine
  constexpr size_t MIN_FRAMES = 64;
  constexpr size_t STATE_SIZE = 4;
  float out_old_left[MIN_FRAMES];
  float out_old_right[MIN_FRAMES];
  double temp_state[STATE_SIZE] = {0};

  float gain = 0.0F;
  const float delta = 1.0F / static_cast<float>(frames);

  size_t count = 0;

  while (count < frames) {
    // compute the block size
    size_t block_size = std::min(MIN_FRAMES, frames - count);

    // Compute the offset
    const float *xl = in_left + count;
    const float *xr = in_right + count;
    float *yl = out_old_left;
    float *yr = out_old_right;
    float *zl = out_left + count;
    float *zr = out_right + count;

    // Process the biquad with old coefficients, discard IIR state
    process_biquad_stereo(old_coeff, temp_state, yl, xl, yr, xr, block_size);

    // Process biquad with new coefficients
    process_biquad_stereo(new_coeff, state, zl, xl, zr, xr, block_size);

    for (size_t i = 0; i < block_size; ++i) {
      zl[i] = ((1.0F - gain) * yl[i]) + (gain * xl[i]);
      zr[i] = ((1.0F - gain) * yr[i]) + (gain * xr[i]);
      gain += delta; // = std::min(delta + gain, 1.0F);
    }

    count += block_size;
  }
}

#endif // A5EQ_DSP_BIQUAD_HPP_
