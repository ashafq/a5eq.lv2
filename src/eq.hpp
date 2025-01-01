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

#ifndef A5EQ_EQ_HPP_
#define A5EQ_EQ_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "eq_math.hpp"
#include "eq_utils.hpp"
#include "iir_filter.hpp"

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

  static constexpr auto XFADE_DURATION_MS = 40.0F;

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

  /**
   * @brief Set the sample rate from host
   */
  void set_sample_rate(float rate) noexcept {
    sample_rate_ = rate;

    xfade_frames_ = static_cast<uint32_t>((rate * XFADE_DURATION_MS * 0.001F) + 0.5F);
    xfade_alpha_ = 0.0F;
    xfade_delta_ = 1.0F / xfade_frames_;

    DBG(printf("Sample rate: %g, Xfade frames: %u, Xfade delta: %g\n", rate,
               xfade_frames_, xfade_delta_););
    update_coefficients();
  }

  /**
   * @brief Connect ports from host
   */
  void connect_port(uint32_t port, void *ptr) noexcept {
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
   * @brief Check if parameters have changed
   */
  void mark_for_transition() noexcept {
    // Check if it is in transition

    DBG(printf("Marking for transition\n"););

    xfade_alpha_ = 0.0F;
  }

  /**
   * @brief Check if parameters have changed
   */
  bool is_in_transition() const noexcept { return xfade_alpha_ < 1.0F; }

  void update_transition() noexcept {
    if (xfade_alpha_ >= 1.0F) {
      // Copy over the tmp_coeff_ to coeff_
      std::copy(tmp_coeff_.begin(), tmp_coeff_.end(), coeff_.begin());

      // Copy the states as well
      std::copy(tmp_state_.begin(), tmp_state_.end(), state_.begin());
    }
  }

  /**
   * @brief Update filter coefficients based on cached parameter values
   */
  void update_coefficients() {
    for (size_t band = 0; band < NUM_BANDS; ++band) {
      if (band == 0) {
        auto coeff =
            compute_low_shelf(cache_freq_control_[band], cache_gain_control_[band],
                              cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else if (band == (NUM_BANDS - 1)) {
        auto coeff =
            compute_high_shelf(cache_freq_control_[band], cache_gain_control_[band],
                               cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else {
        auto coeff = compute_peaking(cache_freq_control_[band], cache_gain_control_[band],
                                     cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      }

      DBG({
        auto b0 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 0];
        auto b1 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 1];
        auto b2 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 2];
        auto a1 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 3];
        auto a2 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 4];
        printf("tmp_coeff_[%zu] = {%g, %g, %g, %g, %g}\n", band, b0, b1, b2, a1, a2);
      });

      mark_for_transition();
    }
  }

  /**
   * @brief Process at enabled state, output buffer gets written with sysout
   *
   * @param frames Number of frames to process
   */
  void process_enabled(uint32_t frames) noexcept {
    auto *input_left = input_left_;
    auto *input_right = input_right_;
    auto *output_left = output_left_;
    auto *output_right = output_right_;

    auto *coeff = coeff_.data();
    auto *state = state_.data();

    // Protect input buffer from all FP exceptions
    fp_buffer_protect_all(input_left, frames);
    fp_buffer_protect_all(input_right, frames);

    if (is_in_transition()) {
      auto *coeff_new = tmp_coeff_.data();
      auto *state_new = tmp_state_.data();

      // Process biquad filter in transition state
      iir_filter_xfade_stereo_process({.in_left = input_left,
                                       .in_right = input_right,

                                       .out_left = output_left,
                                       .out_right = output_right,

                                       .coeff_old = coeff,
                                       .coeff_new = coeff_new,

                                       .state_old = state,
                                       .state_new = state_new,

                                       .n_stages = NUM_BANDS},
                                      xfade_alpha_, xfade_delta_, frames);
      update_transition();
    } else {
      // Process biquad filter in steady state
      iir_filter_stereo_process({.in_left = input_left,
                                 .in_right = input_right,

                                 .out_left = output_left,
                                 .out_right = output_right,

                                 .coeff = coeff,
                                 .state = state,

                                 .n_stages = NUM_BANDS},
                                frames);

      // Check for parameter changes
      check_parameters_chaged();
    }

    // Protect output buffer from all FP exceptions
    fp_buffer_protect_all(output_left, frames);
    fp_buffer_protect_all(output_right, frames);

    // Reset filter state in case of filter instability
    if (fp_buffer_check(state_.data(), state_.size())) {
      state_.fill(0.0F);
    }

    // Reset tmp_state_ buffer in case of filter instability
    if (fp_buffer_check(tmp_state_.data(), tmp_state_.size())) {
      tmp_state_.fill(0.0F);
    }

    return;
  }

  /**
   * @brief Bypassed processing
   */
  void process_bypass(size_t frames) noexcept {
    // Copy left output data if they are different locations in memory
    if (input_left_ != output_left_) {
      std::copy_n(input_left_, frames, output_left_);
    }

    // Similar with the right channel
    if (input_right_ != output_right_) {
      std::copy_n(input_right_, frames, output_right_);
    }
  }

  /**
   * @brief Update the enable state
   */
  void update_enable() noexcept {
    if (enabled_cache_ != (*enabled_)) {
      enabled_cache_prev_ = enabled_cache_;
      enabled_cache_ = (*enabled_);
    }
  }

  /**
   * @brief Main process routine
   *
   * @param frames Number of frames to process
   */
  void process(uint32_t frames) noexcept {
    update_enable();

    // Crossfade between bypass and active states
    if (enabled_cache_ != enabled_cache_prev_) {
      process_enabled(frames);

      // If current_enable_ is true, then system is going from bypass state to
      // active state Fade out input is input buffer
      const auto *input1_left = enabled_cache_ ? output_left_ : input_left_;
      const auto *input1_right = enabled_cache_ ? output_right_ : input_right_;
      // Fade in output is output buffer
      const auto *input2_left = enabled_cache_ ? input_left_ : output_left_;
      const auto *input2_right = enabled_cache_ ? input_right_ : output_right_;

      float *output_left = output_left_;
      float *output_right = output_right_;

      float alpha = 0.0F;
      float delta = 1.0F / frames;

      crossfade_stereo_process({.in1_left = input1_left,
                                .in1_right = input1_right,
                                .in2_left = input2_left,
                                .in2_right = input2_right,
                                .out_left = output_left,
                                .out_right = output_right},
                               alpha, delta, frames);

      enabled_cache_prev_ = enabled_cache_;
    }

    // Process biquad if enable flag is true
    if (enabled_cache_) {
      process_enabled(frames);
    } else {
      process_bypass(frames);
    }
  }

  /**
   * @brief Check if any parameter has changed
   *
   * @return true if any parameter has changed
   * @return false if no parameter has changed
   */
  void check_parameters_chaged() noexcept {
    auto param_changed = false;

    for (size_t band = 0; band < NUM_BANDS; ++band) {
      param_changed |= update_band_coefficients(band);
    }

    if (param_changed) {
      mark_for_transition();
    }
  }

  /**
   * @brief Check if any paramater has changed
   */
  bool update_band_coefficients(size_t band) {
    auto param_changed = false;
    const auto FREQ_MAX = sample_rate_ / 2.0F;

    if (freq_control_[band]) {
      if (!is_close(cache_freq_control_[band], *freq_control_[band], THRESHOLD)) {
        DBG(printf("FREQ[%zu] = %g\n", band, *freq_control_[band]););
        cache_freq_control_[band] = std::clamp(*freq_control_[band], FREQ_MIN, FREQ_MAX);
        param_changed = true;
      }
    }

    if (gain_control_[band]) {
      if (!is_close(cache_gain_control_[band], *gain_control_[band], THRESHOLD)) {
        DBG(printf("GAIN[%zu] = %g\n", band, *gain_control_[band]););
        cache_gain_control_[band] = std::clamp(*gain_control_[band], GAIN_MIN, GAIN_MAX);
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
        auto coeff =
            compute_low_shelf(cache_freq_control_[band], cache_gain_control_[band],
                              cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else if (band == (NUM_BANDS - 1)) {
        auto coeff =
            compute_high_shelf(cache_freq_control_[band], cache_gain_control_[band],
                               cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      } else {
        auto coeff = compute_peaking(cache_freq_control_[band], cache_gain_control_[band],
                                     cache_q_control_[band], sample_rate_);
        std::copy(coeff.begin(), coeff.end(),
                  tmp_coeff_.begin() + (band * NUM_COEFF_PER_BAND));
      }

      DBG({
        auto b0 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 0];
        auto b1 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 1];
        auto b2 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 2];
        auto a1 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 3];
        auto a2 = tmp_coeff_[band * NUM_COEFF_PER_BAND + 4];
        printf("tmp_coeff_[%zu] = {%g, %g, %g, %g, %g}\n", band, b0, b1, b2, a1, a2);
      });
    }

    return param_changed;
  }

private:
  std::array<float *, NUM_BANDS> freq_control_{};
  std::array<float *, NUM_BANDS> gain_control_{};
  std::array<float *, NUM_BANDS> q_control_{};

  std::array<float, NUM_BANDS> cache_freq_control_{160.F, 300.F, 1000.F, 2500.F, 9000.F};

  std::array<float, NUM_BANDS> cache_gain_control_{DEF_GAIN, DEF_GAIN, DEF_GAIN, DEF_GAIN,
                                                   DEF_GAIN};

  std::array<float, NUM_BANDS> cache_q_control_{DEF_Q, DEF_Q, DEF_Q, DEF_Q, DEF_Q};

  // EQ coefficients for each band per channel
  alignas(16) std::array<double, NUM_COEFF_PER_BAND * NUM_BANDS> coeff_{};
  // Temporary coefficients for EQ transition
  alignas(16) std::array<double, NUM_COEFF_PER_BAND * NUM_BANDS> tmp_coeff_{};
  // EQ state for each band per channel
  alignas(16) std::array<double, NUM_STATE_PER_BAND * NUM_BANDS * NUM_CHANNEL> state_{};
  // Temporary state for EQ transition
  alignas(
      16) std::array<double, NUM_STATE_PER_BAND * NUM_BANDS * NUM_CHANNEL> tmp_state_{};

  float sample_rate_{};

  float *input_left_{nullptr};
  float *input_right_{nullptr};
  float *output_left_{nullptr};
  float *output_right_{nullptr};

  uint32_t *enabled_{nullptr};
  uint32_t enabled_cache_{0};
  uint32_t enabled_cache_prev_{0};

  // Counter for crossfading between two states
  uint32_t xfade_frames_{0};
  float xfade_alpha_{0.0F};
  float xfade_delta_{0.0F};
};

#endif // A5EQ_EQ_HPP_
