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

#ifndef A5EQ_EQ_MATH_HPP_
#define A5EQ_EQ_MATH_HPP_

#include "eq_utils.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Compute low shelf eq coefficients
 *
 * @param fc Cutoff/Passband frequency in Hz
 * @param gain Gain in dB
 * @param q Q-factor (Quality or Bandwidth)
 * @param fs Sample rate in Hz
 *
 * @note b[n] are numerator coefficients
 * @note a[n] are denominator coefficients
 *
 * @return std::array<double, 5> Coefficients in {b0, b1, b2, a1, a2}
 */
std::array<double, 5> compute_low_shelf(double fc, double gain, double q, double fs) {
  assert(fs > 0 && "Sample rate must be greater than zero");
  assert(((fc > 0) && (fc <= (fs / 2))) && "Frequency: 0 < fc ");
  assert(q > 0 && "Q must be greater than zero");

  auto w0 = 2.0 * M_PI * (fc / fs);
  auto alpha = std::sin(w0) / (2.0 * q);
  auto A = std::pow(10.0, gain / 40.0);

  // clang-format off
  auto b0 =    A*( (A+1) - ((A-1)*std::cos(w0)) + (2*std::sqrt(A)*alpha));
  auto b1 =  2*A*( (A-1) - ((A+1)*std::cos(w0))                         );
  auto b2 =    A*( (A+1) - ((A-1)*std::cos(w0)) - (2*std::sqrt(A)*alpha));
  auto a0 =        (A+1) + ((A-1)*std::cos(w0)) + (2*std::sqrt(A)*alpha) ;
  auto a1 =   -2*( (A-1) + ((A+1)*std::cos(w0))                         );
  auto a2 =        (A+1) + ((A-1)*std::cos(w0)) - (2*std::sqrt(A)*alpha) ;
  // clang-format on

  DBG({
    printf("%s: fc = %g, gain = %g, Q= %g, fs = %g\n"
           "\t| w0 [%g], alpha [%g], A [%g]\n"
           "\t| %g, %g, %g, %g, %g, %g\n",
           __PRETTY_FUNCTION__, fc, gain, q, fs, w0, alpha, A, b0, b1, b2, a0, a1, a2);
  });

  return {{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}};
}

/**
 * @brief Compute high shelf eq coefficients
 *
 * @param fc Cutoff/Passband frequency in Hz
 * @param gain Gain in dB
 * @param q Q-factor (Quality or Bandwidth)
 * @param fs Sample rate in Hz
 *
 * @note b[n] are numerator coefficients
 * @note a[n] are denominator coefficients
 *
 * @return std::array<double, 5> Coefficients in {b0, b1, b2, a1, a2}
 */
std::array<double, 5> compute_high_shelf(double fc, double gain, double q, double fs) {
  assert(fs > 0 && "Sample rate must be greater than zero");
  assert(((fc > 0) && (fc <= (fs / 2))) && "Frequency: 0 < fc ");
  assert(q > 0 && "Q must be greater than zero");

  auto w0 = 2.0 * M_PI * (fc / fs);
  auto alpha = std::sin(w0) / (2.0 * q);
  auto A = std::pow(10.0, gain / 40.0);

  // clang-format off
  auto b0 =    A*( (A+1) + (A-1)*std::cos(w0) + 2*std::sqrt(A)*alpha );
  auto b1 = -2*A*( (A-1) + (A+1)*std::cos(w0)                        );
  auto b2 =    A*( (A+1) + (A-1)*std::cos(w0) - 2*std::sqrt(A)*alpha );
  auto a0 =        (A+1) - (A-1)*std::cos(w0) + 2*std::sqrt(A)*alpha;
  auto a1 =    2*( (A-1) - (A+1)*std::cos(w0)                        );
  auto a2 =        (A+1) - (A-1)*std::cos(w0) - 2*std::sqrt(A)*alpha;
  // clang-format on

  DBG({
    printf("%s: fc = %g, gain = %g, Q= %g, fs = %g\n"
           "\t| w0 [%g], alpha [%g], A [%g]\n"
           "\t| %g, %g, %g, %g, %g, %g\n",
           __PRETTY_FUNCTION__, fc, gain, q, fs, w0, alpha, A, b0, b1, b2, a0, a1, a2);
  });

  return {{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}};
}

/**
 * @brief Compute peaking eq coefficients
 *
 * @param fc Cutoff/Passband frequency in Hz
 * @param gain Gain in dB
 * @param q Q-factor (Quality or Bandwidth)
 * @param fs Sample rate in Hz
 *
 * @note b[n] are numerator coefficients
 * @note a[n] are denominator coefficients
 *
 * @return std::array<double, 5> Coefficients in {b0, b1, b2, a1, a2}
 */
std::array<double, 5> compute_peaking(double fc, double gain, double q, double fs) {
  assert(fs > 0 && "Sample rate must be greater than zero");
  assert(((fc > 0) && (fc <= (fs / 2))) && "Frequency: 0 < fc ");
  assert(q > 0 && "Q must be greater than zero");

  auto w0 = 2.0 * M_PI * (fc / fs);
  auto alpha = std::sin(w0) / (2.0 * q);
  auto A = std::pow(10.0, gain / 40.0);

  auto b0 = 1.0 + alpha * A;
  auto b1 = -2.0 * std::cos(w0);
  auto b2 = 1.0 - alpha * A;
  auto a0 = 1.0 + alpha / A;
  auto a1 = -2.0 * std::cos(w0);
  auto a2 = 1.0 - alpha / A;

  DBG({
    printf("%s: fc = %g, gain = %g, Q= %g, fs = %g\n"
           "\t| w0 [%g], alpha [%g], A [%g]\n"
           "\t| %g, %g, %g, %g, %g, %g\n",
           __PRETTY_FUNCTION__, fc, gain, q, fs, w0, alpha, A, b0, b1, b2, a0, a1, a2);
  });

  return {{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}};
}

#endif // A5EQ_EQ_MATH_HPP_