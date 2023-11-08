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

#include <cstddef>

#if (__SSE__ == 1)
#include <xmmintrin.h>
#endif

#if (__FMA__ == 1)
#include <immintrin.h>
#endif

/************************
 * Filter implementations
 ************************/

#if (__FMA__ == 1)
#warning "Using FMA optimized biquad filter"

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
static inline void process_biquad_stereo(const double *coeff, double *state,
                                         float *out_left, const float *in_left,
                                         float *out_right,
                                         const float *in_right, size_t frames) {
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

    // Process 1nd pair

    // y0 = (b0 * x0) + s1
    __m128d y0 = _mm_fmadd_pd(b0, x0, s1);
    // s1 = (b1 * x0) + (a1 * y0) + s2
    s1 = _mm_fmadd_pd(b1, x0, _mm_fmadd_pd(a1, y0, s2));
    // s2 = (b2 * x0) + (a2 * y0)
    s2 = _mm_fmadd_pd(b2, x0, _mm_mul_pd(a2, y0));

    // Process 2nd pair

    // y1 = (b0 * x0) + s1
    __m128d y1 = _mm_fmadd_pd(b0, x1, s1);
    // s1 = (b1 * x1) + (a1 * y1) + s2
    s1 = _mm_fmadd_pd(b1, x1, _mm_fmadd_pd(a1, y1, s2));
    // s2 = (b2 * x1) + (a2 * y1)
    s2 = _mm_fmadd_pd(b2, x1, _mm_mul_pd(a2, y1));

    // Process 3nd pair

    // y2 = (b0 * x2) + s1
    __m128d y2 = _mm_fmadd_pd(b0, x2, s1);
    // s1 = (b1 * x2) + (a1 * y2) + s2
    s1 = _mm_fmadd_pd(b1, x2, _mm_fmadd_pd(a1, y2, s2));
    // s2 = (b2 * x2) + (a2 * y2)
    s2 = _mm_fmadd_pd(b2, x2, _mm_mul_pd(a2, y2));

    // Process 4nd pair

    // y3 = (b0 * x3) + s1
    __m128d y3 = _mm_fmadd_pd(b0, x3, s1);
    // s1 = (b1 * x3) + (a1 * y3) + s2
    s1 = _mm_fmadd_pd(b1, x3, _mm_fmadd_pd(a1, y3, s2));
    // s2 = (b2 * x3) + (a2 * y3)
    s2 = _mm_fmadd_pd(b2, x3, _mm_mul_pd(a2, y3));

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

    // printf("count = %zu\n", count);
    // printf("simd_frames = %zu\n", simd_frames);
  }

  // Process any remaining frames
  while (count < frames) {
    // Load input samples into SSE registers and convert to double
    __m128d x_left = _mm_cvtps_pd(_mm_load_ss(in_left + count));
    __m128d x_right = _mm_cvtps_pd(_mm_load_ss(in_right + count));

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left, x_right);

    // y0 = (b0 * x0) + s1
    __m128d y0 = _mm_fmadd_pd(b0, x0, s1);
    // s1 = (b1 * x0) + (a1 * y0) + s2
    s1 = _mm_fmadd_pd(b1, x0, _mm_fmadd_pd(a1, y0, s2));
    // s2 = (b2 * x0) + (a2 * y0)
    s2 = _mm_fmadd_pd(b2, x0, _mm_mul_pd(a2, y0));

    // y0 = {yl, yr}
    __m128d yl = _mm_unpacklo_pd(y0, y0); // {yl, yl}
    __m128d yr = _mm_unpackhi_pd(y0, y0); // {yr, yr}

    // Store the result into out_left and out_right
    _mm_store_ss(out_left + count, _mm_cvtpd_ps(yl));
    _mm_store_ss(out_right + count, _mm_cvtpd_ps(yr));

    // printf("count = %zu\n", count);
    // printf("frames = %zu\n", frames);

    ++count;
  }

  // Store states
  _mm_storeu_pd(state + 0, s1);
  _mm_storeu_pd(state + 2, s2);
}


#elif (__SSE__ == 1)
#warning "Using SSE optimized biquad filter"

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
static inline void process_biquad_stereo(const double *coeff, double *state,
                                         float *out_left, const float *in_left,
                                         float *out_right,
                                         const float *in_right, size_t frames) {
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

    // Process 1nd pair

    // y0 = (b0 * x0) + s1
    __m128d y0 = _mm_add_pd(_mm_mul_pd(b0, x0), s1);
    // s1 = (b1 * x0) + (a1 * y0) + s2
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x0), _mm_mul_pd(a1, y0)), s2);
    // s2 = (b2 * x0) + (a2 * y0)
    s2 = _mm_add_pd(_mm_mul_pd(b2, x0), _mm_mul_pd(a2, y0));

    // Process 2nd pair

    // y1 = (b0 * x1) + s1
    __m128d y1 = _mm_add_pd(_mm_mul_pd(b0, x1), s1);
    // s1 = (b1 * x1) + (a1 * y1) + s2
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x1), _mm_mul_pd(a1, y1)), s2);
    // s2 = (b2 * x1) + (a2 * y1)
    s2 = _mm_add_pd(_mm_mul_pd(b2, x1), _mm_mul_pd(a2, y1));

    // Process 3nd pair

    // y2 = (b0 * x2) + s1
    __m128d y2 = _mm_add_pd(_mm_mul_pd(b0, x2), s1);
    // s1 = (b1 * x2) + (a1 * y2) + s2
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x2), _mm_mul_pd(a1, y2)), s2);
    // s2 = (b2 * x2) + (a2 * y2)
    s2 = _mm_add_pd(_mm_mul_pd(b2, x2), _mm_mul_pd(a2, y2));

    // Process 4nd pair

    // y3 = (b0 * x3) + s1
    __m128d y3 = _mm_add_pd(_mm_mul_pd(b0, x3), s1);
    // s1 = (b1 * x3) + (a1 * y3) + s2
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x3), _mm_mul_pd(a1, y3)), s2);
    // s2 = (b2 * x3) + (a2 * y3)
    s2 = _mm_add_pd(_mm_mul_pd(b2, x3), _mm_mul_pd(a2, y3));

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

    // printf("count = %zu\n", count);
    // printf("simd_frames = %zu\n", simd_frames);
  }

  // Process any remaining frames
  while (count < frames) {
    // Load input samples into SSE registers and convert to double
    __m128d x_left = _mm_cvtps_pd(_mm_load_ss(in_left + count));
    __m128d x_right = _mm_cvtps_pd(_mm_load_ss(in_right + count));

    // Interleave left and right inputs for stereo pair
    __m128d x0 = _mm_unpacklo_pd(x_left, x_right);

    // y0 = (b0 * x0) + s1
    __m128d y0 = _mm_add_pd(_mm_mul_pd(b0, x0), s1);
    // s1 = (b1 * x0) + (a1 * y0) + s2
    s1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(b1, x0), _mm_mul_pd(a1, y0)), s2);
    // s2 = (b2 * x0) + (a2 * y0)
    s2 = _mm_add_pd(_mm_mul_pd(b2, x0), _mm_mul_pd(a2, y0));

    // y0 = {yl, yr}
    __m128d yl = _mm_unpacklo_pd(y0, y0); // {yl, yl}
    __m128d yr = _mm_unpackhi_pd(y0, y0); // {yr, yr}

    // Store the result into out_left and out_right
    _mm_store_ss(out_left + count, _mm_cvtpd_ps(yl));
    _mm_store_ss(out_right + count, _mm_cvtpd_ps(yr));

    // printf("count = %zu\n", count);
    // printf("frames = %zu\n", frames);

    ++count;
  }

  // Store states
  _mm_storeu_pd(state + 0, s1);
  _mm_storeu_pd(state + 2, s2);
}

#else // __SSE__
#warning "Using generic implementation of biquad"

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

#endif // __AVX2__, __SSE__, generic

/**
 * @brief Process biquad filter for a single stage of biquad and transition
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
static void process_biquad_stereo_xfade(const double *old_coeff,
                                        const double *new_coeff, double *state,
                                        float *out_left, const float *in_left,
                                        float *out_right, const float *in_right,
                                        size_t frames) {
  // Load old coefficients
  const auto b0_old = *old_coeff++;
  const auto b1_old = *old_coeff++;
  const auto b2_old = *old_coeff++;
  const auto a1_old = *old_coeff++;
  const auto a2_old = *old_coeff++;

  // Load new coefficients
  const auto b0_new = *new_coeff++;
  const auto b1_new = *new_coeff++;
  const auto b2_new = *new_coeff++;
  const auto a1_new = *new_coeff++;
  const auto a2_new = *new_coeff++;

  // Process left channel
  auto s1_old = state[0];
  auto s2_old = state[2];

  auto s1_new = 0.0;
  auto s2_new = 0.0;

  auto gain = 0.0;
  const auto delta = 1.0 / frames;

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_left[i]);

    auto out_old = (b0_old * in) + s1_old;
    s1_old = (b1_old * in) - (a1_old * out_old) + s2_old;
    s2_old = (b2_old * in) - (a2_old * out_old);

    auto out_new = (b0_new * in) + s1_new;
    s1_new = (b1_new * in) - (a1_new * out_new) + s2_new;
    s2_new = (b2_new * in) - (a2_new * out_new);

    auto out = ((1.0 - gain) * out_old) + (gain * out_new);

    gain += delta;

    out_left[i] = static_cast<float>(out);
  }

  state[0] = s1_new;
  state[2] = s2_new;

  // Process right channel
  s1_old = state[1];
  s2_old = state[3];

  s1_new = 0.0;
  s2_new = 0.0;

  gain = 0.0;

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_right[i]);

    auto out_old = (b0_old * in) + s1_old;
    s1_old = (b1_old * in) - (a1_old * out_old) + s2_old;
    s2_old = (b2_old * in) - (a2_old * out_old);

    auto out_new = (b0_new * in) + s1_new;
    s1_new = (b1_new * in) - (a1_new * out_new) + s2_new;
    s2_new = (b2_new * in) - (a2_new * out_new);

    auto out = ((1.0 - gain) * out_old) + (gain * out_new);

    gain += delta;

    out_right[i] = static_cast<float>(out);
  }

  state[1] = s1_new;
  state[3] = s2_new;
}

#endif // A5EQ_DSP_BIQUAD_HPP_
