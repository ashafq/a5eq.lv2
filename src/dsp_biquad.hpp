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
  auto s2 = state[1];

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_left[i]);

    auto out = (b0 * in) + s1;
    s1 = (b1 * in) - (a1 * out) + s2;
    s2 = (b2 * in) - (a2 * out);

    out_left[i] = static_cast<float>(out);
  }

  state[0] = s1;
  state[1] = s2;

  // Process right channel
  s1 = state[2];
  s2 = state[3];

  for (size_t i = 0; i < frames; ++i) {
    auto in = static_cast<double>(in_right[i]);

    auto out = (b0 * in) + s1;
    s1 = (b1 * in) - (a1 * out) + s2;
    s2 = (b2 * in) - (a2 * out);

    out_right[i] = static_cast<float>(out);
  }

  state[2] = s1;
  state[3] = s2;
}

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
  auto s2_old = state[1];

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
  state[1] = s2_new;

  // Process right channel
  s1_old = state[2];
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

  state[2] = s1_new;
  state[3] = s2_new;
}

#endif // A5EQ_DSP_BIQUAD_HPP_