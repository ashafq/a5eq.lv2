/*
 * Copyright (C) 2025 Ayan Shafqat <ayan.x.shafqat@gmail.com>
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

#ifndef A5EQ_DSP_CROSSFADE_HPP_
#define A5EQ_DSP_CROSSFADE_HPP_

#include <algorithm>

#include "dsp_biquad.hpp"

static inline float dsp_cubic_fadeout(float t) {
  // Horner's method for:  1 - 3t^2 + 2t^3
  return 1.0F - (t * t) * (3.0F - (2.0F * t));
}

typedef struct dsp_crossfade_stereo_process {
  const float *in1_left;  //!< Pointer to the first input left channel, fading out
  const float *in1_right; //!< Pointer to the first input right channel, fading out
  const float *in2_left;  //!< Pointer to the second input left channel, fading in
  const float *in2_right; //!< Pointer to the second input right channel, fading in
  float *out_left;        //!< Pointer to the output left channel
  float *out_right;       //!< Pointer to the output right channel
} dsp_crossfade_stereo_process_t;

#if (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION) ||                     \
    (DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_SSE2_OPTIMIZATION)

/**
 * @brief Cubic fadeout using SIMD
 * @param[in] t Input value [0.0F, 1.0F]
 * @return __m128 output value
 */
static inline __m128 dsp_cubic_fadeout_ps(__m128 t) {
  // Horner's method for:
  // 1 - 3t^2 + 2t^3 = 1.0F - (t * t) * (3.0F - (2.0F * t))

  // Let x2 = t^2
  // Let x3 = 3 - 2 * t
  // let y = 1 - x2 * x3

#if DSP_BIQAD_OPTIMIZATION == DSP_BIQUAD_AVX2_FMA_OPTIMIZATION

  // Use FMA for__m128
  __m128 x2 = _mm_mul_ps(t, t);
  __m128 x3 = _mm_fnmadd_ps(_mm_set1_ps(2.0F), t, _mm_set1_ps(3.0F));
  return _mm_fnmadd_ps(x2, x3, _mm_set1_ps(1.0F));

#else // SSE2

  __m128 x2 = _mm_mul_ps(t, t);
  __m128 x3 = _mm_sub_ps(_mm_set1_ps(3.0F), _mm_mul_ps(_mm_set1_ps(2.0F), t));
  return _mm_sub_ps(_mm_set1_ps(1.0F), _mm_mul_ps(x2, x3));

#endif
}

/**
 * @brief Process stereo audio samples using crossfade
 *
 * This function processes stereo audio samples using crossfade. The function
 * processes each stereo frame of audio samples using the specified alpha and
 * delta values. The function processes the audio samples in place.
 *
 * @param[in] buffer Pointer to the crossfade filter structure
 * @param[in,out] alpha Alpha value, crossfade state
 * @param[in] delta Delta value, change in Alpha per step
 * @param[in] frames Number of stereo frames
 *
 * @return float Alpha value, updated crossfade state
 */
static float dsp_crossfade_stereo_process(const dsp_crossfade_stereo_process_t &buffer,
                                          float &alpha, float delta, size_t frames) {
  size_t simd_frames = frames & (~static_cast<size_t>(3));
  size_t count = 0;

  alignas(sizeof(__m128)) float alpha_values[4] = {alpha, alpha + delta,
                                                   alpha + 2 * delta, alpha + 3 * delta};

  __m128 alpha_ps = _mm_load_ps(alpha_values);
  __m128 delta_ps = _mm_set1_ps(delta);

  while (count < simd_frames) {
    // Load alpha values
    __m128 fade_out_gain = dsp_cubic_fadeout_ps(alpha_ps);
    __m128 fade_in_gain = _mm_sub_ps(_mm_set1_ps(1.0F), fade_out_gain);

    // Load input samples
    __m128 in1_left = _mm_loadu_ps(buffer.in1_left + count);
    __m128 in1_right = _mm_loadu_ps(buffer.in1_right + count);
    __m128 in2_left = _mm_loadu_ps(buffer.in2_left + count);
    __m128 in2_right = _mm_loadu_ps(buffer.in2_right + count);

    // Process the samples
    __m128 out_left = _mm_add_ps(_mm_mul_ps(fade_out_gain, in1_left),
                                 _mm_mul_ps(fade_in_gain, in2_left));
    __m128 out_right = _mm_add_ps(_mm_mul_ps(fade_out_gain, in1_right),
                                  _mm_mul_ps(fade_in_gain, in2_right));

    // Store the output
    _mm_storeu_ps(buffer.out_left + count, out_left);
    _mm_storeu_ps(buffer.out_right + count, out_right);

    // Update alpha values
    alpha_ps = _mm_add_ps(alpha_ps, delta_ps);
    alpha_ps = _mm_min_ps(alpha_ps, _mm_set1_ps(1.0F));

    count += 4;
  }

  // Get the upper alpha value
  alpha = _mm_cvtss_f32(_mm_shuffle_ps(alpha_ps, alpha_ps, _MM_SHUFFLE(3, 3, 3, 3)));

  // Process any remaining frames
  while (count < frames) {
    float fade_out_gain = dsp_cubic_fadeout(alpha);
    float fade_in_gain = 1.0F - fade_out_gain;

    float in1_left = buffer.in1_left[count];
    float in1_right = buffer.in1_right[count];
    float in2_left = buffer.in2_left[count];
    float in2_right = buffer.in2_right[count];

    buffer.out_left[count] = (fade_out_gain * in1_left) + (fade_in_gain * in2_left);
    buffer.out_right[count] = (fade_out_gain * in1_right) + (fade_in_gain * in2_right);

    alpha += delta;
    alpha = std::min<float>(alpha, 1.0F);

    ++count;
  }

  return alpha;
}

#elif DSP_BIQUAD_OPTIMIZATION == DSP_BIQUAD_ARM_NEON_OPTIMIZATION

/**
 * @brief Cubic fadeout using SIMD
 * @param[in] t Input value [0.0F, 1.0F]
 * @return float32x4_t output value
 */
static inline float32x4_t dsp_cubic_fadeout_ps(float32x4_t t) {
  // Horner's method for:
  // 1 - 3t^2 + 2t^3 = 1.0F - (t * t) * (3.0F - (2.0F * t))

  // Let x2 = t^2
  // Let x3 = 3 - 2 * t
  // let y = 1 - x2 * x3

  // Use FMA for__m128
  float32x4_t x2 = vmulq_f32(t, t);
  float32x4_t x3 = vmlsq_f32(vdupq_n_f32(3.0F), t, vdupq_n_f32(2.0F));
  return vmlsq_f32(vdupq_n_f32(1.0F), x2, x3);
}

/**
 * @brief Process stereo audio samples using crossfade
 *
 * This function processes stereo audio samples using crossfade. The function
 * processes each stereo frame of audio samples using the specified alpha and
 * delta values. The function processes the audio samples in place.
 *
 * @param[in] buffer Pointer to the crossfade filter structure
 * @param[in,out] alpha Alpha value, crossfade state
 * @param[in] delta Delta value, change in Alpha per step
 * @param[in] frames Number of stereo frames
 *
 * @return float Alpha value, updated crossfade state
 */
static float dsp_crossfade_stereo_process(const dsp_crossfade_stereo_process_t &buffer,
                                          float &alpha, float delta, size_t frames) {
  size_t simd_frames = frames & (~static_cast<size_t>(3));
  size_t count = 0;

  alignas(sizeof(float32x4_t)) float alpha_values[4] = {
      alpha, alpha + delta, alpha + 2 * delta, alpha + 3 * delta};

  float32x4_t alpha_ps = vld1q_f32(alpha_values);
  float32x4_t delta_ps = vdupq_n_f32(delta);

  while (count < simd_frames) {
    // Load alpha values
    float32x4_t fade_out_gain = dsp_cubic_fadeout_ps(alpha_ps);
    float32x4_t fade_in_gain = vsubq_f32(vdupq_n_f32(1.0F), fade_out_gain);

    // Load input samples
    float32x4_t in1_left = vld1q_f32(buffer.in1_left + count);
    float32x4_t in1_right = vld1q_f32(buffer.in1_right + count);
    float32x4_t in2_left = vld1q_f32(buffer.in2_left + count);
    float32x4_t in2_right = vld1q_f32(buffer.in2_right + count);

    // Process the samples
    float32x4_t out_left =
        vaddq_f32(vmulq_f32(fade_out_gain, in1_left), vmulq_f32(fade_in_gain, in2_left));
    float32x4_t out_right = vaddq_f32(vmulq_f32(fade_out_gain, in1_right),
                                      vmulq_f32(fade_in_gain, in2_right));

    // Store the output
    vst1q_f32(buffer.out_left + count, out_left);
    vst1q_f32(buffer.out_right + count, out_right);

    // Update alpha values
    alpha_ps = vaddq_f32(alpha_ps, delta_ps);
    alpha_ps = vminq_f32(alpha_ps, vdupq_n_f32(1.0F));

    count += 4;
  }

  // Get the upper most register for alpha value
  alpha = vgetq_lane_f32(alpha_ps, 3);

  // Process any remaining frames
  while (count < frames) {
    float fade_out_gain = dsp_cubic_fadeout(alpha);
    float fade_in_gain = 1.0F - fade_out_gain;

    float in1_left = buffer.in1_left[count];
    float in1_right = buffer.in1_right[count];
    float in2_left = buffer.in2_left[count];
    float in2_right = buffer.in2_right[count];

    buffer.out_left[count] = (fade_out_gain * in1_left) + (fade_in_gain * in2_left);
    buffer.out_right[count] = (fade_out_gain * in1_right) + (fade_in_gain * in2_right);

    alpha += delta;
    alpha = std::min<float>(alpha, 1.0F);

    ++count;
  }

  return alpha;
}
#else // Generic implementation

/**
 * @brief Process stereo audio samples using crossfade
 *
 * This function processes stereo audio samples using crossfade. The function
 * processes each stereo frame of audio samples using the specified alpha and
 * delta values. The function processes the audio samples in place.
 *
 * @param[in] buffer Pointer to the crossfade filter structure
 * @param[in,out] alpha Alpha value, crossfade state
 * @param[in] delta Delta value, change in Alpha per step
 * @param[in] frames Number of stereo frames
 *
 * @return float Alpha value, updated crossfade state
 */
static float dsp_crossfade_stereo_process(const dsp_crossfade_stereo_process_t &buffer,
                                          float &alpha, float delta, size_t frames) {
  for (size_t i = 0; i < frames; ++i) {
    float fade_out_gain = dsp_cubic_fadeout(alpha);
    float fade_in_gain = 1.0F - fade_out_gain;

    float in1_left = buffer.in1_left[i];
    float in1_right = buffer.in1_right[i];
    float in2_left = buffer.in2_left[i];
    float in2_right = buffer.in2_right[i];

    buffer.out_left[i] = (fade_out_gain * in1_left) + (fade_in_gain * in2_left);
    buffer.out_right[i] = (fade_out_gain * in1_right) + (fade_in_gain * in2_right);

    alpha += delta;
    alpha = std::min<float>(alpha, 1.0F);
  }

  return alpha;
}

#endif // DSP_BIQUAD_OPTIMIZATION

static float crossfade_stereo_process(const dsp_crossfade_stereo_process_t &buffer,
                                      float &alpha, float delta, size_t frames) {
  // Process stereo audio samples using crossfade in blocks
  alignas(DSP_BIQUAD_ALIGNMENT) float null_buffer[DSP_BIQUAD_MIN_FRAME_SIZE];

  std::memset(null_buffer, 0, sizeof(null_buffer));

  // Process the audio samples in blocks of DSP_BIQUAD_MIN_FRAME_SIZE
  size_t count = 0;
  while (count < frames) {
    size_t frames_to_process =
        std::min<size_t>(frames - count, DSP_BIQUAD_MIN_FRAME_SIZE);

    // Null check for input and output buffers
    const float *in1_left =
        (buffer.in1_left == nullptr) ? null_buffer : (buffer.in1_left + count);
    const float *in1_right =
        (buffer.in1_right == nullptr) ? null_buffer : (buffer.in1_right + count);
    const float *in2_left =
        (buffer.in2_left == nullptr) ? null_buffer : (buffer.in2_left + count);
    const float *in2_right =
        (buffer.in2_right == nullptr) ? null_buffer : (buffer.in2_right + count);
    float *out_left =
        (buffer.out_left == nullptr) ? null_buffer : (buffer.out_left + count);
    float *out_right =
        (buffer.out_right == nullptr) ? null_buffer : (buffer.out_right + count);

    dsp_crossfade_stereo_process({.in1_left = in1_left,
                                  .in1_right = in1_right,
                                  .in2_left = in2_left,
                                  .in2_right = in2_right,
                                  .out_left = out_left,
                                  .out_right = out_right},
                                 alpha, delta, frames_to_process);

    count += frames_to_process;
  }

  return alpha;
}
#endif // A5EQ_DSP_CROSSFADE_HPP_
