/*
 * Copyright (C) 2023-2025 Ayan Shafqat <ayan.x.shafqat@gmail.com>
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
#include <cstring>

#include "eq_utils.hpp"

#define DSP_BIQUAD_ARM_NEON_OPTIMIZATION (3)
#define DSP_BIQUAD_AVX2_FMA_OPTIMIZATION (2)
#define DSP_BIQUAD_SSE2_OPTIMIZATION (1)
#define DSP_BIQUAD_NO_OPTIMIZATION (0)

#define DSP_BIQUAD_NUM_COEFFS_PER_STAGE (5)
#define DSP_BIQUAD_NUM_STATE_PER_STAGE (2)
#define DSP_BIQUAD_MIN_FRAME_SIZE (128)
#define DSP_BIQUAD_MAX_STAGES (16)
#define DSP_BIQUAD_ALIGNMENT (16)

#if defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED(ptr)                                                              \
  static_cast<decltype(ptr)>(__builtin_assume_aligned(ptr, (DSP_BIQUAD_ALIGNMENT)))
#else
#define ASSUME_ALIGNED(ptr) (ptr)
#endif

#if (defined(__aarch64__)) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
// Include NEON intrinsic header file
#include <arm_neon.h>
#define DSP_BIQUAD_OPTIMIZATION (DSP_BIQUAD_ARM_NEON_OPTIMIZATION)
#elif defined(__x86_64__) && defined(__FMA__)
// Include FMA intrinsic header file
#include <immintrin.h>
#define DSP_BIQUAD_OPTIMIZATION (DSP_BIQUAD_AVX2_FMA_OPTIMIZATION)
#elif defined(__x86_64__) && defined(__SSE2__)
// Include SSE+ intrinsic header file
#include <xmmintrin.h>
#define DSP_BIQUAD_OPTIMIZATION (DSP_BIQUAD_SSE2_OPTIMIZATION)
#else
#define DSP_BIQUAD_OPTIMIZATION (DSP_BIQUAD_NO_OPTIMIZATION)
#endif // Select optimization levels

#ifdef DSP_BIQUAD_MESSAGE_ENABLED
#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_ARM_NEON_OPTIMIZATION)
#pragma message "Compiling with ARM/Aarch64 NEON optimizations"
#elif (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION)
#pragma message "Compiling with x86-64 with AVX2 FMA optimizations"
#elif (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_SSE2_OPTIMIZATION)
#pragma message "Compiling with x86-64 with SSE2 optimizations"
#else
#pragma message "Compiling with no specific optimizations"
#endif // Select optimization levels
#endif // DSP_BIQUAD_MESSAGE_ENABLED

/************************
 * Filter implementations
 ************************/

/**
 * PROCESS_BIQUAD notes
 *
 * This macro implments transposed direct form II biquad filter using double
 * precision floating point operations. The macro processes a single stereo
 * frame of audio samples.
 *
 * The macro executes the following operations:
 *
 * y[n]   = b0 * x[n] + w[n-1]
 * w[n]   = b1 * x[n] + a1 * y[n] + w[n-1]
 * w[n-1] = b2 * x[n] + a2 * y[n]
 *
 * The macro takes the following arguments:
 *
 * x0: The input sample to be processed
 * y0: The output sample (stereo frame)
 * b0: The feedforward coefficient b0
 * b1: The feedforward coefficient b1
 * b2: The feedforward coefficient b2
 * a1: The feedback coefficient a1, negative
 * a2: The feedback coefficient a2, negative
 * s1: The state variable s1 (w[n-1])
 * s2: The state variable s2 (w[n-2])
 */

#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_ARM_NEON_OPTIMIZATION)

// The PROCESS_BIQUAD macro would be defined for NEON
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                               \
  do {                                                                                   \
    /* y0 = (b0 * x0) + s1 */                                                            \
    y0 = vmlaq_f64(s1, b0, x0);                                                          \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                                \
    s1 = vmlaq_f64(s2, b1, x0);                                                          \
    s1 = vmlaq_f64(s1, a1, y0);                                                          \
    /* s2 = (b2 * x0) + (a2 * y0) */                                                     \
    s2 = vmlaq_f64(vmulq_f64(a2, y0), b2, x0);                                           \
  } while (false)

#elif (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION)

// The PROCESS_BIQUAD macro would be defined for AVX2
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                               \
  do {                                                                                   \
    /* y0 = (b0 * x0) + s1 */                                                            \
    y0 = _mm_fmadd_pd(b0, x0, s1);                                                       \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                                \
    s1 = _mm_fmadd_pd(b1, x0, _mm_fmadd_pd(a1, y0, s2));                                 \
    /* s2 = (b2 * x0) + (a2 * y0) */                                                     \
    s2 = _mm_fmadd_pd(b2, x0, _mm_mul_pd(a2, y0));                                       \
  } while (false)

#elif (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_SSE2_OPTIMIZATION)

// The PROCESS_BIQUAD macro would be defined for SSE2
#define PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2)                               \
  do {                                                                                   \
    /* y0 = (b0 * x0) + s1 */                                                            \
    y0 = _mm_add_pd(_mm_mul_pd(b0, x0), s1);                                             \
    /* s1 = (b1 * x0) + (a1 * y0) + s2 */                                                \
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x0), _mm_mul_pd(a1, y0)), s2);             \
    /* s2 = (b2 * x0) + (a2 * y0) */                                                     \
    s2 = _mm_add_pd(_mm_mul_pd(b2, x0), _mm_mul_pd(a2, y0));                             \
  } while (false)

#endif // Select optimization levels

/**
 * Intrinsics for stereo processing of biquad filters optimized for AMD64
 */

#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION) ||                     \
    (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_SSE2_OPTIMIZATION)

/**
 * @brief Process stereo audio samples using a cascade of biquad filters
 *
 * This function processes stereo audio samples using a cascade of biquad
 * filters. The function processes each stereo frame of audio samples using the
 * specified number of biquad stages. The function processes the audio samples
 * in place.
 *
 * @param[in] coeff Pointer to the array of biquad coefficients
 * @param[in,out] state Pointer to the array of biquad state variables
 * @param[in] input Pointer to the array of input stereo audio samples
 * @param[out] output Pointer to the array of output stereo audio samples
 * @param[in] n_stages Number of biquad stages
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_tdf2_cascade_biquad_process_stereo(const double *coeff, double *state,
                                                   const double *input, double *output,
                                                   size_t n_stages, size_t n_frames) {
  size_t coeff_offset = 0;
  size_t state_offset = 0;

  state = ASSUME_ALIGNED(state);
  input = ASSUME_ALIGNED(input);
  output = ASSUME_ALIGNED(output);

  for (size_t stage = 0; stage < n_stages; ++stage) {
    // Load coeffs and states
    __m128d b0 = _mm_set1_pd(+coeff[coeff_offset + 0]);
    __m128d b1 = _mm_set1_pd(+coeff[coeff_offset + 1]);
    __m128d b2 = _mm_set1_pd(+coeff[coeff_offset + 2]);
    __m128d a1 = _mm_set1_pd(-coeff[coeff_offset + 3]);
    __m128d a2 = _mm_set1_pd(-coeff[coeff_offset + 4]);

    __m128d s1 = _mm_load_pd(state + state_offset + 0);
    __m128d s2 = _mm_load_pd(state + state_offset + 2);

    const double *in_ptr = (stage == 0) ? input : output;
    double *out_ptr = output;

    for (size_t frame = 0; frame < n_frames; ++frame) {
      size_t offset = frame * 2;

      // Load the input
      __m128d x0 = _mm_load_pd(in_ptr + offset);
      __m128d y0 = _mm_setzero_pd();

      PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);

      // Store the output
      _mm_store_pd(out_ptr + offset, y0);
    }

    // Save states
    _mm_store_pd(state + state_offset + 0, s1);
    _mm_store_pd(state + state_offset + 2, s2);

    // Update offsets
    coeff_offset += DSP_BIQUAD_NUM_COEFFS_PER_STAGE;

    // Each stage has two state variables over two channels
    state_offset += (DSP_BIQUAD_NUM_STATE_PER_STAGE * 2);
  }

  return;
}

/**
 * @brief Interleave two float arrays into a double array
 *
 * @param[in] in_left Pointer to the array of left channel samples
 * @param[in] in_right Pointer to the array of right channel samples
 * @param[out] out Pointer to the array of interleaved stereo samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_interleave_f32x2_f64(const float *in_left, const float *in_right,
                                     double *out, size_t n_frames) {
  out = ASSUME_ALIGNED(out);

  size_t simd_frames = n_frames & (~static_cast<size_t>(3));
  size_t count = 0;

  while (count < simd_frames) {
    // Compute offsets: input is non-interleaved, and output is interleaved
    size_t in_offset = count;
    size_t out_offset = 2 * count;

    // Load input samples into SSE registers and convert to double
    __m128 x_left = _mm_loadu_ps(in_left + in_offset);   // {l0, l1, l2, l3}
    __m128 x_right = _mm_loadu_ps(in_right + in_offset); // {r0, r1, r2, r3}

    __m128d x_left_0 = _mm_cvtps_pd(x_left);                        // {l0, l1} (double)
    __m128d x_right_0 = _mm_cvtps_pd(x_right);                      // {r0, r1} (double)
    __m128d x_left_1 = _mm_cvtps_pd(_mm_movehl_ps(x_left, x_left)); // {l2, l3} (double)
    __m128d x_right_1 =
        _mm_cvtps_pd(_mm_movehl_ps(x_right, x_right)); // {r2, r3} (double)

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left_0, x_right_0); // {l0, r0}
    __m128d x1 = _mm_unpackhi_pd(x_left_0, x_right_0); // {l1, r1}
    __m128d x2 = _mm_unpacklo_pd(x_left_1, x_right_1); // {l2, r2}
    __m128d x3 = _mm_unpackhi_pd(x_left_1, x_right_1); // {l3, r3})

    // Store interleaved stereo pairs
    _mm_store_pd(out + out_offset + 0, x0);
    _mm_store_pd(out + out_offset + 2, x1);
    _mm_store_pd(out + out_offset + 4, x2);
    _mm_store_pd(out + out_offset + 6, x3);

    count += 4;
  }

  // Process any remaining frames
  while (count < n_frames) {
    // Compute offsets: input is non-interleaved, and output is interleaved
    size_t in_offset = count;
    size_t out_offset = 2 * count;

    // Load input samples into SSE registers and convert to double
    __m128d x_left = _mm_cvtps_pd(_mm_load_ss(in_left + in_offset));
    __m128d x_right = _mm_cvtps_pd(_mm_load_ss(in_right + in_offset));

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left, x_right);

    // Store interleaved stereo pairs
    _mm_store_pd(out + out_offset, x0);

    ++count;
  }

  return;
}

/**
 * @brief De-interleave a double array into two float arrays
 *
 * @param[in] in Pointer to the array of interleaved stereo samples
 * @param[out] out_left Pointer to the array of left channel samples
 * @param[out] out_right Pointer to the array of right channel samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_deinterleave_f64_f32x2(const double *in, float *out_left,
                                       float *out_right, size_t n_frames) {
  in = ASSUME_ALIGNED(in);

  size_t simd_frames = n_frames & (~static_cast<size_t>(3));
  size_t count = 0;

  while (count < simd_frames) {
    // Compute offsets: input is interleaved, and output is non-interleaved
    size_t in_offset = 2 * count;
    size_t out_offset = count;

    // Load the input samples
    __m128d y0 = _mm_load_pd(in + in_offset + 0); // {l0, r0}
    __m128d y1 = _mm_load_pd(in + in_offset + 2); // {l1, r1}
    __m128d y2 = _mm_load_pd(in + in_offset + 4); // {l2, r2}
    __m128d y3 = _mm_load_pd(in + in_offset + 6); // {l3, r3}

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

    // Store the output
    _mm_storeu_ps(out_left + out_offset, y_left);
    _mm_storeu_ps(out_right + out_offset, y_right);

    count += 4;
  }

  // Process any remaining frames
  while (count < n_frames) {
    // Compute offsets: input is interleaved, and output is non-interleaved
    size_t in_offset = 2 * count;
    size_t out_offset = count;

    __m128d y0 = _mm_load_pd(in + in_offset + 0); // {l0, r0}

    // y0 = {yl, yr}
    __m128d yl = _mm_unpacklo_pd(y0, y0); // {yl, yl}
    __m128d yr = _mm_unpackhi_pd(y0, y0); // {yr, yr}

    // Store the result into out_left and out_right
    _mm_store_ss(out_left + out_offset, _mm_cvtpd_ps(yl));
    _mm_store_ss(out_right + out_offset, _mm_cvtpd_ps(yr));

    ++count;
  }

  return;
}
#endif // (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION) ||
       // (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_SSE2_OPTIMIZATION)

/**
 * Intrinsics for stereo processing of biquad filters optimized for ARM64
 */
#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_ARM_NEON_OPTIMIZATION)

/**
 * @brief Process stereo audio samples using a cascade of biquad filters
 *
 * This function processes stereo audio samples using a cascade of biquad
 * filters. The function processes each stereo frame of audio samples using the
 * specified number of biquad stages. The function processes the audio samples
 * in place.
 *
 * @param[in] coeff Pointer to the array of biquad coefficients
 * @param[in,out] state Pointer to the array of biquad state variables
 * @param[in] input Pointer to the array of input stereo audio samples
 * @param[out] output Pointer to the array of output stereo audio samples
 * @param[in] n_stages Number of biquad stages
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_tdf2_cascade_biquad_process_stereo(const double *coeff, double *state,
                                                   const double *input, double *output,
                                                   size_t n_stages, size_t n_frames) {
  size_t coeff_offset = 0;
  size_t state_offset = 0;

  state = ASSUME_ALIGNED(state);
  input = ASSUME_ALIGNED(input);
  output = ASSUME_ALIGNED(output);

  for (size_t stage = 0; stage < n_stages; ++stage) {
    // Load coeffs and states
    float64x2_t b0 = vdupq_n_f64(+coeff[coeff_offset + 0]);
    float64x2_t b1 = vdupq_n_f64(+coeff[coeff_offset + 1]);
    float64x2_t b2 = vdupq_n_f64(+coeff[coeff_offset + 2]);
    float64x2_t a1 = vdupq_n_f64(-coeff[coeff_offset + 3]);
    float64x2_t a2 = vdupq_n_f64(-coeff[coeff_offset + 4]);

    float64x2_t s1 = vld1q_f64(state + state_offset + 0);
    float64x2_t s2 = vld1q_f64(state + state_offset + 2);

    const double *in_ptr = (stage == 0) ? input : output;
    double *out_ptr = output;

    for (size_t frame = 0; frame < n_frames; ++frame) {
      size_t offset = frame * 2;

      // Load the input
      float64x2_t x0 = vld1q_f64(in_ptr + offset);
      float64x2_t y0 = vdupq_n_f64(0.0);

      PROCESS_BIQUAD(x0, y0, b0, b1, b2, a1, a2, s1, s2);

      // Store the output
      vst1q_f64(out_ptr + offset, y0);
    }

    // Save states
    vst1q_f64(state + state_offset + 0, s1);
    vst1q_f64(state + state_offset + 2, s2);

    // Update offsets
    coeff_offset += DSP_BIQUAD_NUM_COEFFS_PER_STAGE;

    // Each stage has two state variables over two channels
    state_offset += (DSP_BIQUAD_NUM_STATE_PER_STAGE * 2);
  }

  return;
}

/**
 * @brief Interleave two float arrays into a double array
 *
 * @param[in] in_left Pointer to the array of left channel samples
 * @param[in] in_right Pointer to the array of right channel samples
 * @param[out] out Pointer to the array of interleaved stereo samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_interleave_f32x2_f64(const float *in_left, const float *in_right,
                                     double *out, size_t n_frames) {
  out = ASSUME_ALIGNED(out);

  size_t simd_frames = n_frames & (~static_cast<size_t>(3));
  size_t count = 0;

  while (count < simd_frames) {
    // Compute offsets: input is non-interleaved, and output is interleaved
    size_t in_offset = count;
    size_t out_offset = 2 * count;

    // Load input samples into SSE registers and convert to double
    float32x4_t x_left = vld1q_f32(in_left + in_offset);   // {l0, l1, l2, l3}
    float32x4_t x_right = vld1q_f32(in_right + in_offset); // {l0, l1, l2, l3}

    float64x2_t x_left_0 = vcvt_f64_f32(vget_low_f32(x_left));   // {l0, l1} (double)
    float64x2_t x_right_0 = vcvt_f64_f32(vget_low_f32(x_right)); // {r0, r1} (double)

    float64x2_t x_left_1 = vcvt_f64_f32(vget_high_f32(x_left));   // {l2, l3} (double)
    float64x2_t x_right_1 = vcvt_f64_f32(vget_high_f32(x_right)); // {r2, r3} (double)

    // Interleave left and right inputs for stereo pair
    float64x2_t x0 = vtrn1q_f64(x_left_0, x_right_0); // {l0, r0}
    float64x2_t x1 = vtrn2q_f64(x_left_0, x_right_0); // {l1, r1}
    float64x2_t x2 = vtrn1q_f64(x_left_1, x_right_1); // {l2, r2}
    float64x2_t x3 = vtrn2q_f64(x_left_1, x_right_1); // {l3, r3}

    // Store interleaved stereo pairs
    vst1q_f64(out + out_offset + 0, x0);
    vst1q_f64(out + out_offset + 2, x1);
    vst1q_f64(out + out_offset + 4, x2);
    vst1q_f64(out + out_offset + 6, x3);

    count += 4;
  }

  // Process any remaining frames
  while (count < n_frames) {
    // Compute offsets: input is non-interleaved, and output is interleaved
    size_t in_offset = count;
    size_t out_offset = 2 * count;

    // Load single float values and create float32x2_t vectors
    float64x1_t x_left = vdup_n_f64(static_cast<double>(in_left[in_offset]));
    float64x1_t x_right = vdup_n_f64(static_cast<double>(in_right[in_offset]));

    // Interleave left and right inputs for stereo pair
    float64x2_t x0 = vcombine_f64(x_left, x_right);

    // Store interleaved stereo pairs
    vst1q_f64(out + out_offset, x0);

    ++count;
  }

  return;
}

/**
 * @brief De-interleave a double array into two float arrays
 *
 * @param[in] in Pointer to the array of interleaved stereo samples
 * @param[out] out_left Pointer to the array of left channel samples
 * @param[out] out_right Pointer to the array of right channel samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_deinterleave_f64_f32x2(const double *in, float *out_left,
                                       float *out_right, size_t n_frames) {
  in = ASSUME_ALIGNED(in);

  size_t simd_frames = n_frames & (~static_cast<size_t>(3));
  size_t count = 0;

  while (count < simd_frames) {
    // Compute offsets: input is interleaved, and output is non-interleaved
    size_t in_offset = 2 * count;
    size_t out_offset = count;

    // Load the input samples
    float64x2_t y0 = vld1q_f64(in + in_offset + 0); // {l0, r0}
    float64x2_t y1 = vld1q_f64(in + in_offset + 2); // {l1, r1}
    float64x2_t y2 = vld1q_f64(in + in_offset + 4); // {l2, r2}
    float64x2_t y3 = vld1q_f64(in + in_offset + 6); // {l3, r3}

    // De-interleave the pairs
    float64x2_t y_left_0 = vuzp1q_f64(y0, y1);  // {l0, l1}
    float64x2_t y_right_0 = vuzp2q_f64(y0, y1); // {r0, r1}
    float64x2_t y_left_1 = vuzp1q_f64(y2, y3);  // {l2, l3}
    float64x2_t y_right_1 = vuzp2q_f64(y2, y3); // {r2, r3}

    // Convert the output to float and combine them into two registers
    float32x4_t y_left =
        vcombine_f32(vcvt_f32_f64(y_left_0), vcvt_f32_f64(y_left_1)); // {l0, l1, l2, l3}
    float32x4_t y_right = vcombine_f32(vcvt_f32_f64(y_right_0),
                                       vcvt_f32_f64(y_right_1)); // {r0, r1, r2, r3}

    // Store the output
    vst1q_f32(out_left + out_offset, y_left);
    vst1q_f32(out_right + out_offset, y_right);

    count += 4;
  }

  // Process any remaining frames
  while (count < n_frames) {
    // Compute offsets: input is interleaved, and output is non-interleaved
    size_t in_offset = 2 * count;
    size_t out_offset = count;

    float64x2_t y0 = vld1q_f64(in + in_offset + 0); // {l0, r0}

    // y0 = {yl, yr}
    float64x1_t yl = vget_low_f64(y0);  // {yl}
    float64x1_t yr = vget_high_f64(y0); // {yr}

    // Store the results
    out_left[out_offset] = static_cast<float32_t>(vget_lane_f64(yl, 0));
    out_right[out_offset] = static_cast<float32_t>(vget_lane_f64(yr, 0));

    ++count;
  }

  return;
}

#endif // (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_ARM_NEON_OPTIMIZATION)

#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_NO_OPTIMIZATION)
/**
 * @brief Process stereo audio samples using a cascade of biquad filters
 *
 * This function processes stereo audio samples using a cascade of biquad
 * filters. The function processes each stereo frame of audio samples using the
 * specified number of biquad stages. The function processes the audio samples
 * in place.
 *
 * @param[in] coeff Pointer to the array of biquad coefficients
 * @param[in,out] state Pointer to the array of biquad state variables
 * @param[in] input Pointer to the array of input stereo audio samples
 * @param[out] output Pointer to the array of output stereo audio samples
 * @param[in] n_stages Number of biquad stages
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_tdf2_cascade_biquad_process_stereo(const double *coeff, double *state,
                                                   const double *input, double *output,
                                                   size_t n_stages, size_t n_frames) {
  size_t coeff_offset = 0;
  size_t state_offset = 0;

  for (size_t stage = 0; stage < n_stages; ++stage) {
    // Load coeffs and states
    const double b0 = +coeff[coeff_offset + 0];
    const double b1 = +coeff[coeff_offset + 1];
    const double b2 = +coeff[coeff_offset + 2];
    const double a1 = -coeff[coeff_offset + 3];
    const double a2 = -coeff[coeff_offset + 4];

    double s1_left = state[state_offset + 0];
    double s1_right = state[state_offset + 1];
    double s2_left = state[state_offset + 2];
    double s2_right = state[state_offset + 3];

    const double *in_ptr = (stage == 0) ? input : output;
    double *out_ptr = output;

    for (size_t frame = 0; frame < n_frames; ++frame) {
      size_t offset = frame * 2;

      // Process left channel
      double x0_left = in_ptr[offset];
      double y0_left = (b0 * x0_left) + s1_left;
      s1_left = (b1 * x0_left) + (a1 * y0_left) + s2_left;
      s2_left = (b2 * x0_left) + (a2 * y0_left);

      // Process right channel
      double x0_right = in_ptr[offset + 1];
      double y0_right = (b0 * x0_right) + s1_right;
      s1_right = (b1 * x0_right) + (a1 * y0_right) + s2_right;
      s2_right = (b2 * x0_right) + (a2 * y0_right);

      out_ptr[offset] = y0_left;
      out_ptr[offset + 1] = y0_right;
    }

    // Save states
    state[state_offset + 0] = s1_left;
    state[state_offset + 1] = s1_right;
    state[state_offset + 2] = s2_left;
    state[state_offset + 3] = s2_right;

    // Update offsets
    coeff_offset += DSP_BIQUAD_NUM_COEFFS_PER_STAGE;

    // Each stage has two state variables over two channels
    state_offset += (DSP_BIQUAD_NUM_STATE_PER_STAGE * 2);
  }
}

/**
 * @brief Interleave two float arrays into a double array
 *
 * @param[in] in_left Pointer to the array of left channel samples
 * @param[in] in_right Pointer to the array of right channel samples
 * @param[out] out Pointer to the array of interleaved stereo samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_interleave_f32x2_f64(const float *in_left, const float *in_right,
                                     double *out, size_t n_frames) {
  for (size_t frame = 0; frame < n_frames; ++frame) {
    size_t offset = frame * 2;
    out[offset] = static_cast<double>(in_left[frame]);
    out[offset + 1] = static_cast<double>(in_right[frame]);
  }
}

/**
 * @brief De-interleave a double array into two float arrays
 *
 * @param[in] in Pointer to the array of interleaved stereo samples
 * @param[out] out_left Pointer to the array of left channel samples
 * @param[out] out_right Pointer to the array of right channel samples
 * @param[in] n_frames Number of stereo frames
 */
static void dsp_deinterleave_f64_f32x2(const double *in, float *out_left,
                                       float *out_right, size_t n_frames) {
  for (size_t frame = 0; frame < n_frames; ++frame) {
    size_t offset = frame * 2;
    out_left[frame] = static_cast<float>(in[offset]);
    out_right[frame] = static_cast<float>(in[offset + 1]);
  }
}

#endif // (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_NO_OPTIMIZATION)

#undef PROCESS_BIQUAD // Undefine the PROCESS_BIQUAD macro

#endif // A5EQ_DSP_BIQUAD_HPP_
