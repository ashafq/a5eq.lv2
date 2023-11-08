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

#ifndef A5EQ_EQ_HPP_
#define A5EQ_EQ_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "eq_math.hpp"
#include "eq_utils.hpp"
#include "dsp_biquad.hpp"

/**
 * @brief Main EQ class
 *
 * Topology:
 *
 * Low Shelf --> Peaking --> Peaking --> Peaking --> High Shelf
 */
class A5Eq {
public:
  static constexpr size_t NUM_CHANNEL = 2;
  static constexpr size_t NUM_BANDS = 5;
  static constexpr size_t NUM_COEFF_PER_BAND = 5;
  static constexpr size_t NUM_STATE_PER_BAND = 4;
  static constexpr size_t MINIMUM_FRAME_SIZE = 64;

  static constexpr auto THRESHOLD = 1.0e-3F;
  static constexpr auto FREQ_MIN = 20.0F;
  static constexpr auto GAIN_MIN = -20.0F;
  static constexpr auto GAIN_MAX = +20.0F;
  static constexpr auto Q_MIN = 0.1F;
  static constexpr auto Q_MAX = 4.0F;

  static constexpr auto DEF_GAIN = 0.0F;
  static constexpr auto DEF_Q = 0.3F;

  enum Port : uint32_t {
    FREQ_PORT_START = 0,
    FREQ_PORT_END = FREQ_PORT_START + NUM_BANDS,
    GAIN_PORT_START = FREQ_PORT_END,
    GAIN_PORT_END = GAIN_PORT_START + NUM_BANDS,
    Q_PORT_START = GAIN_PORT_END,
    Q_PORT_END = Q_PORT_START + NUM_BANDS,
    SYS_ENABLE_PORT = Q_PORT_END,
    INPUT_LEFT = SYS_ENABLE_PORT + 1,
    OUTPUT_LEFT = SYS_ENABLE_PORT + 2,
    INPUT_RIGHT = SYS_ENABLE_PORT + 3,
    OUTPUT_RIGHT = SYS_ENABLE_PORT + 4,
  };

  A5Eq() = default;
  ~A5Eq() = default;

  void set_sample_rate(float rate) {
    sample_rate_ = rate;
    update_coefficients();
  }

  void connect_port(uint32_t port, void *ptr) {
    if ((port >= FREQ_PORT_START) && (port < FREQ_PORT_END)) {
      DBG(printf("FREQ[%u] = %p\n", port - FREQ_PORT_START, ptr););
      freq_control_[port - FREQ_PORT_START] = static_cast<float *>(ptr);
    } else if ((port >= GAIN_PORT_START) && (port < GAIN_PORT_END)) {
      DBG(printf("GAIN[%u] = %p\n", port - GAIN_PORT_START, ptr););
      gain_control_[port - GAIN_PORT_START] = static_cast<float *>(ptr);
    } else if ((port >= Q_PORT_START) && (port < Q_PORT_END)) {
      DBG(printf("Q[%u] = %p\n", port - Q_PORT_START, ptr););
      q_control_[port - Q_PORT_START] = static_cast<float *>(ptr);
    } else if (port == SYS_ENABLE_PORT) {
      DBG(printf("SYS_ENABLE = %p\n", ptr););
      enabled_ = static_cast<uint32_t *>(ptr);
    } else if (port == INPUT_LEFT) {
      DBG(printf("INPUT_LEFT = %p\n", ptr););
      input_left_ = static_cast<float *>(ptr);
    } else if (port == OUTPUT_LEFT) {
      DBG(printf("OUTPUT_LEFT = %p\n", ptr););
      output_left_ = static_cast<float *>(ptr);
    } else if (port == INPUT_RIGHT) {
      DBG(printf("INPUT_RIGHT = %p\n", ptr););
      input_right_ = static_cast<float *>(ptr);
    } else if (port == OUTPUT_RIGHT) {
      DBG(printf("OUTPUT_RIGHT = %p\n", ptr););
      output_right_ = static_cast<float *>(ptr);
    } else {
      DBG(printf("INVALID = %p\n", ptr););
    }
  }

  /**
   * @brief Update filter coefficients based on cached parameter values
   */
  void update_coefficients() {
    for (size_t band = 0; band < NUM_BANDS; ++band) {
      if (band == 0) {
        auto coeff = compute_low_shelf(cache_freq_control_[band],
                                       cache_gain_control_[band],
                                       cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else if (band == (NUM_BANDS - 1)) {
        auto coeff = compute_high_shelf(cache_freq_control_[band],
                                        cache_gain_control_[band],
                                        cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else {
        auto coeff = compute_peaking(cache_freq_control_[band],
                                     cache_gain_control_[band],
                                     cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      }

      DBG({
        auto b0 = coeff_[band * NUM_COEFF_PER_BAND + 0];
        auto b1 = coeff_[band * NUM_COEFF_PER_BAND + 1];
        auto b2 = coeff_[band * NUM_COEFF_PER_BAND + 2];
        auto a1 = coeff_[band * NUM_COEFF_PER_BAND + 3];
        auto a2 = coeff_[band * NUM_COEFF_PER_BAND + 4];
        printf("coeff[%zu] = {%g, %g, %g, %g, %g}\n", band, b0, b1, b2, a1, a2);
      });
    }
  }

  /**
   * @brief Main process routine
   * @param frames Number of samples to process
   */
  void process(uint32_t frames) {
    auto enabled = *enabled_;

    // Process biquad if enable flag is true
    if (enabled) {
      uint32_t processed_frames = 0;

      while (processed_frames < frames) {
        // Select the right buffers for processing
        auto *input_left = input_left_ ? (input_left_ + processed_frames)
                                       : null_buffer_.data();
        auto *input_right = input_right_ ? (input_right_ + processed_frames)
                                         : null_buffer_.data();
        auto *output_left = output_left_ ? (output_left_ + processed_frames)
                                         : null_buffer_.data();
        auto *output_right = output_right_ ? (output_right_ + processed_frames)
                                           : null_buffer_.data();

        auto n_proc = std::min(frames - processed_frames,
                               static_cast<uint32_t>(MINIMUM_FRAME_SIZE));

        // Protect input buffer from all FP exceptions
        fp_buffer_protect_all(input_left, n_proc);
        fp_buffer_protect_all(input_right, n_proc);

        // Process biquad bands
        for (uint32_t band = 0; band < NUM_BANDS; ++band) {
          const auto *coeff = coeff_.data() + (band * NUM_COEFF_PER_BAND);
          auto *state =
              state_.data() + (band * NUM_STATE_PER_BAND * NUM_CHANNEL);

          auto *xl = (band == 0) ? input_left : output_left;
          auto *yl = output_left;
          auto *xr = (band == 0) ? input_right : output_right;
          auto *yr = output_right;

          if (parameters_changed(band)) {
            auto *new_coeff = tmp_coeff_.data();
            process_biquad_stereo_xfade(coeff, new_coeff, state, yl, xl, yr, xr,
                                        n_proc);
            std::copy(tmp_coeff_.begin(), tmp_coeff_.end(),
                      coeff_.begin() + (band * NUM_COEFF_PER_BAND));
          } else {
            process_biquad_stereo(coeff, state, yl, xl, yr, xr, n_proc);
          }
        }

        // Protect output buffer from all FP exceptions
        fp_buffer_protect_all(output_left, n_proc);
        fp_buffer_protect_all(output_right, n_proc);

        // Do the next iteration
        processed_frames += n_proc;
      }

      // Reset filter state in case of filter instability
      if (fp_buffer_check(state_.data(), state_.size())) {
        state_.fill(0.0F);
      }

      // Protect state memory from going into denormal region, which
      // reduces processing throughput
      fp_buffer_protect_subnormal(state_.data(), state_.size());
    } else {
      // Copy left output data
      if (input_left_ != output_left_) {
        memcpy(output_left_, input_left_, frames * sizeof(float));
      }
      // Copy right output data
      if (input_right_ != output_right_) {
        memcpy(output_right_, input_right_, frames * sizeof(float));
      }
    }
  }

  bool parameters_changed(size_t band) {
    auto param_changed = false;
    const auto FREQ_MAX = sample_rate_ / 2.0F;

    if (freq_control_[band]) {
      if (!is_close(cache_freq_control_[band], *freq_control_[band],
                    THRESHOLD)) {
        DBG(printf("FREQ[%zu] = %g\n", band, *freq_control_[band]););
        cache_freq_control_[band] =
            std::clamp(*freq_control_[band], FREQ_MIN, FREQ_MAX);
        param_changed = true;
      }
    }

    if (gain_control_[band]) {
      if (!is_close(cache_gain_control_[band], *gain_control_[band],
                    THRESHOLD)) {
        DBG(printf("GAIN[%zu] = %g\n", band, *gain_control_[band]););
        cache_gain_control_[band] =
            std::clamp(*gain_control_[band], GAIN_MIN, GAIN_MAX);
        param_changed = true;
      }
    }

    if (q_control_[band]) {
      if (!is_close(cache_q_control_[band], *q_control_[band], THRESHOLD)) {
        DBG(printf("Q[%zu] = %g\n", band, *q_control_[band]););
        cache_q_control_[band] = std::clamp(*q_control_[band], Q_MIN, Q_MAX);
        param_changed = true;
      }
    }

    // Update the coefficient if parameter has changed
    if (param_changed) {
      if (band == 0) {
        tmp_coeff_ = compute_low_shelf(cache_freq_control_[band],
                                       cache_gain_control_[band],
                                       cache_q_control_[band], sample_rate_);
      } else if (band == (NUM_BANDS - 1)) {
        tmp_coeff_ = compute_high_shelf(cache_freq_control_[band],
                                        cache_gain_control_[band],
                                        cache_q_control_[band], sample_rate_);
      } else {
        tmp_coeff_ = compute_peaking(cache_freq_control_[band],
                                     cache_gain_control_[band],
                                     cache_q_control_[band], sample_rate_);
      }

      DBG({
        auto b0 = tmp_coeff_[0];
        auto b1 = tmp_coeff_[1];
        auto b2 = tmp_coeff_[2];
        auto a1 = tmp_coeff_[3];
        auto a2 = tmp_coeff_[4];
        printf("coeff[%zu] = {%g, %g, %g, %g, %g}\n", band, b0, b1, b2, a1, a2);
      });
    }

    return param_changed;
  }

private:
  std::array<float *, NUM_BANDS> freq_control_{};
  std::array<float *, NUM_BANDS> gain_control_{};
  std::array<float *, NUM_BANDS> q_control_{};

  std::array<float, NUM_BANDS> cache_freq_control_{160.F, 300.F, 1000.F, 2500.F,
                                                   9000.F};

  std::array<float, NUM_BANDS> cache_gain_control_{DEF_GAIN, DEF_GAIN, DEF_GAIN,
                                                   DEF_GAIN, DEF_GAIN};

  std::array<float, NUM_BANDS> cache_q_control_{DEF_Q, DEF_Q, DEF_Q, DEF_Q,
                                                DEF_Q};

  std::array<double, NUM_COEFF_PER_BAND * NUM_BANDS> coeff_{};
  // Temporary coefficients for EQ transition
  std::array<double, NUM_COEFF_PER_BAND> tmp_coeff_{};

  std::array<double, NUM_STATE_PER_BAND * NUM_BANDS * NUM_CHANNEL> state_{};

  std::array<float, MINIMUM_FRAME_SIZE> null_buffer_{};

  float sample_rate_{};

  float *input_left_{nullptr};
  float *input_right_{nullptr};
  float *output_left_{nullptr};
  float *output_right_{nullptr};

  uint32_t *enabled_{nullptr};
};

#endif // A5EQ_EQ_HPP_
