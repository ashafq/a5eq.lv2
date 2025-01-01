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

#ifndef A5EQ_EQ_UTILS_H_
#define A5EQ_EQ_UTILS_H_

#include <cmath>
#include <cstddef>

#ifdef DEBUG_ENABLE

#define DBG(X)                                                                           \
  do {                                                                                   \
    X                                                                                    \
  } while (0)
#define DBGx(x) (void)(0)

#else
#define DBG(X) (void)(0)
#define DBGx(x) (void)(0)

#endif // DBG

/**
 * @brief Compare @p a and @p b if they are close to each other
 *
 * @param a A number
 * @param b Another number
 * @param th Threshold
 *
 * @return true If they are equal or close
 * @return false If they are not equal or close
 */
template <typename DataType> bool is_close(DataType a, DataType b, DataType th) {
  // Check for infinity and 0.0F
  if (a == b) {
    return true;
  } else {
    return std::abs(a - b) <= th;
  }
}

/**
 * @brief Check if any of the buffer is in non-finite region of FP
 *
 * @param[in] src Source/input buffer
 * @param len Number of samples in buffer
 *
 * @return true There is a single sample in buffer that is in
 *    non-finite region
 * @return false Buffer is "all-good"
 */
template <typename DataType> bool fp_buffer_check(const DataType *src, size_t len) {
  if (!src) {
    return true;
  }

  for (size_t i = 0; i < len; ++i) {
    if (!std::isfinite(src[i])) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Protect buffer against INF, NAN, and subnormal values
 *
 * @param[out] dst Output buffer
 * @param len Number of samples to process
 */
template <typename DataType> void fp_buffer_protect_all(DataType *dst, size_t len) {
  if (!dst) {
    return;
  }

  for (size_t i = 0; i < len; ++i) {
    auto x = dst[i];
    if (!(std::isfinite(x) && std::isnormal(x))) {
      dst[i] = static_cast<DataType>(0);
    }
  }
}

/**
 * @brief Protect buffer against subnormal values only
 *
 * @param[out] dst Output buffer
 * @param len Number of samples to process
 */
template <typename DataType> void fp_buffer_protect_subnormal(DataType *dst, size_t len) {
  if (!dst) {
    return;
  }

  for (size_t i = 0; i < len; ++i) {
    if (!std::isnormal(dst[i])) {
      dst[i] = static_cast<DataType>(0);
    }
  }
}

#endif // A5EQ_EQ_UTILS_H_