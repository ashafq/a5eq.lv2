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

#ifndef A5EQ_DSP_IIR_FILTER_HPP_
#define A5EQ_DSP_IIR_FILTER_HPP_

#include "dsp_biquad.hpp"
#include "dsp_crossfade.hpp"

/**
 * @brief Cascade of biquad filter structure for stereo audio samples
 */
typedef struct tdf2_cascade_biquad_process {
  const float *in_left{nullptr};
  const float *in_right{nullptr};

  float *out_left{nullptr};
  float *out_right{nullptr};

  double *input_buffer{nullptr};
  double *output_buffer{nullptr};

  double *coeff{nullptr};
  double *state{nullptr};
  size_t n_stages{0};
} tdf2_cascade_biquad_process_t;

/**
 * @brief Internal function to process stereo audio samples using a cascade of
 * biquad filters
 *
 * This is an internal function that is shared with the rest of the modules
 *
 * @param[in] param Pointer to the biquad filter structure
 * @param[in] frames_to_process Number of stereo frames to process
 *
 */
static void tdf2_cascade_biquad_process_stereo(const tdf2_cascade_biquad_process_t &param,
                                               size_t frames_to_process) {
  auto in_left = param.in_left;
  auto *in_right = param.in_right;
  auto *out_left = param.out_left;
  auto *out_right = param.out_right;

  auto *input_buffer = param.input_buffer;
  auto *output_buffer = param.output_buffer;

  auto *coeff = param.coeff;
  auto *state = param.state;
  auto n_stages = param.n_stages;

  // Interleave the input samples
  dsp_interleave_f32x2_f64(in_left, in_right, input_buffer, frames_to_process);

  // Process the interleaved samples
  dsp_tdf2_cascade_biquad_process_stereo(coeff, state, input_buffer, output_buffer,
                                         n_stages, frames_to_process);

  // De-interleave the output samples
  dsp_deinterleave_f64_f32x2(output_buffer, out_left, out_right, frames_to_process);
}

/**
 * @brief IIR filter structure for stereo audio samples
 */
typedef struct iir_filter_stereo {
  const float *in_left{nullptr};
  const float *in_right{nullptr};

  float *out_left{nullptr};
  float *out_right{nullptr};

  double *coeff{nullptr};
  double *state{nullptr};

  size_t n_stages{0};
} iir_filter_stereo_t;

/**
 * @brief Process stereo audio samples using a cascade of biquad filters
 *
 * This function processes stereo audio samples using a cascade of biquad
 * filters. The function processes each stereo frame of audio samples using the
 * specified number of biquad stages. The function processes the audio samples
 * in place.
 *
 * @param[in] filter Pointer to the biquad filter structure
 * @param[in] n_frames Number of stereo frames
 */
static void iir_filter_stereo_process(const iir_filter_stereo_t &filter,
                                      size_t n_frames) {
  alignas(DSP_BIQUAD_ALIGNMENT) static float null_buffer[DSP_BIQUAD_MIN_FRAME_SIZE];

  alignas(DSP_BIQUAD_ALIGNMENT) double input_buffer[2 * DSP_BIQUAD_MIN_FRAME_SIZE];
  alignas(DSP_BIQUAD_ALIGNMENT) double output_buffer[2 * DSP_BIQUAD_MIN_FRAME_SIZE];

  double *coeff = (ASSUME_ALIGNED(filter.coeff));
  double *state = ASSUME_ALIGNED(filter.state);

  // Null check for coefficients and state
  if (!coeff || !state) {
    DBG({
      printf("%s: Null pointer detected: (coeff: %p, state: %p)\n", __func__,
             static_cast<const void *>(coeff), static_cast<void *>(state));
    });
    return;
  }

  size_t n_stages = filter.n_stages;
  size_t frame = 0;

  memset(null_buffer, 0, sizeof(null_buffer));

  while (frame < n_frames) {
    size_t frames_to_process =
        std::min<size_t>(n_frames - frame, DSP_BIQUAD_MIN_FRAME_SIZE);

    // Null check for input and output buffers
    const float *in_left =
        (filter.in_left == nullptr) ? null_buffer : (filter.in_left + frame);
    const float *in_right =
        (filter.in_right == nullptr) ? null_buffer : (filter.in_right + frame);
    float *out_left =
        (filter.out_left == nullptr) ? null_buffer : (filter.out_left + frame);
    float *out_right =
        (filter.out_right == nullptr) ? null_buffer : (filter.out_right + frame);

    tdf2_cascade_biquad_process_stereo(
        {
            .in_left = in_left,
            .in_right = in_right,

            .out_left = out_left,
            .out_right = out_right,

            .input_buffer = input_buffer,
            .output_buffer = output_buffer,

            .coeff = coeff,
            .state = state,

            .n_stages = n_stages,
        },
        frames_to_process);

    // Update the frame counter
    frame += frames_to_process;
  }

  return;
}

/**
 * @brief IIR filter structure for stereo audio samples
 */
typedef struct iir_filter_xfade_stereo {
  const float *in_left{nullptr};
  const float *in_right{nullptr};

  float *out_left{nullptr};
  float *out_right{nullptr};

  double *coeff_old{nullptr};
  double *coeff_new{nullptr};

  double *state_old{nullptr};
  double *state_new{nullptr};

  size_t n_stages{0};
} iir_filter_xfade_stereo_t;

/**
 * @brief Process stereo audio samples using a crossfade between two sets of
 * biquad filters
 *
 * @param[in] param Pointer to the biquad filter structure
 * @param[in] n_frames Number of stereo frames
 */
static void iir_filter_xfade_stereo_process(const iir_filter_xfade_stereo_t &param,
                                            float &alpha, float delta, size_t n_frames) {
  alignas(DSP_BIQUAD_ALIGNMENT) static float null_buffer[DSP_BIQUAD_MIN_FRAME_SIZE];

  alignas(DSP_BIQUAD_ALIGNMENT) static float temp_out_left[DSP_BIQUAD_MIN_FRAME_SIZE];
  alignas(DSP_BIQUAD_ALIGNMENT) static float temp_out_right[DSP_BIQUAD_MIN_FRAME_SIZE];

  alignas(DSP_BIQUAD_ALIGNMENT) double input_buffer[2 * DSP_BIQUAD_MIN_FRAME_SIZE];
  alignas(DSP_BIQUAD_ALIGNMENT) double output_buffer[2 * DSP_BIQUAD_MIN_FRAME_SIZE];

  // Null check for coefficients and state
  if (!param.coeff_old || !param.coeff_new || !param.state_old || !param.state_new) {
    DBG({
      printf("%s: Null pointer detected: (coeff_old: %p, coeff_new: %p, "
             "state_old: %p, state_new: %p)\n",
             __func__, static_cast<void *>(param.coeff_old),
             static_cast<void *>(param.coeff_new), static_cast<void *>(param.state_old),
             static_cast<void *>(param.state_new));
    });
    return;
  }

  size_t frame = 0;

  // Set null buffer to zero
  std::memset(null_buffer, 0, sizeof(null_buffer));

  // Process the audio samples
  while (frame < n_frames) {
    size_t frames_to_process =
        std::min<size_t>(n_frames - frame, DSP_BIQUAD_MIN_FRAME_SIZE);

    // Null check for input and output buffers
    const float *in_left =
        (param.in_left == nullptr) ? null_buffer : (param.in_left + frame);
    const float *in_right =
        (param.in_right == nullptr) ? null_buffer : (param.in_right + frame);
    float *out_left =
        (param.out_left == nullptr) ? null_buffer : (param.out_left + frame);
    float *out_right =
        (param.out_right == nullptr) ? null_buffer : (param.out_right + frame);

    // Process the old filter
    tdf2_cascade_biquad_process_stereo(
        {
            .in_left = in_left,
            .in_right = in_right,

            .out_left = out_left,
            .out_right = out_right,

            .input_buffer = input_buffer,
            .output_buffer = output_buffer,

            .coeff = param.coeff_old,
            .state = param.state_old,

            .n_stages = param.n_stages,
        },
        frames_to_process);

    // Process the new filter
    tdf2_cascade_biquad_process_stereo(
        {
            .in_left = in_left,
            .in_right = in_right,

            .out_left = temp_out_left,
            .out_right = temp_out_right,

            .input_buffer = input_buffer,
            .output_buffer = output_buffer,

            .coeff = param.coeff_new,
            .state = param.state_new,

            .n_stages = param.n_stages,
        },
        frames_to_process);

    // Crossfade the output
    dsp_crossfade_stereo_process(
        {
            .in1_left = out_left,
            .in1_right = out_right,

            .in2_left = temp_out_left,
            .in2_right = temp_out_right,

            .out_left = out_left,
            .out_right = out_right,
        },
        alpha, delta, frames_to_process);

    frame += frames_to_process;
  }

  return;
}

#endif // A5EQ_DSP_IIR_FILTER_HPP_