#ifndef __TEST_H__
#define __TEST_H__

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define init_linear(a, n_)            \
  do {                                \
    int64_t _n_ = n_;                 \
    for (int64_t i = 0; i < _n_; ++i) \
      a[i] = (i + 1);                 \
  } while (0)

#define init_odd(a, n_)               \
  do {                                \
    int64_t _n_ = n_;                 \
    for (int64_t i = 0; i < _n_; ++i) \
      a[i] = (i * 2 + 1);             \
  } while (0)

#define init_even(a, n_)              \
  do {                                \
    int64_t _n_ = n_;                 \
    for (int64_t i = 0; i < _n_; ++i) \
      a[i] = (i + 1) * 2;             \
  } while (0)

#define init_rand(a, n_)              \
  do {                                \
    int64_t _n_ = n_;                 \
    for (int64_t i = 0; i < _n_; ++i) \
      a[i] = rand() % _n_;            \
  } while (0)

#define compare(a, b, n_, fmt)            \
  do {                                    \
    int64_t _n_ = n_;                     \
    for (int64_t i = 0; i < _n_; ++i) {   \
      if (1e-5 < (a[i] - b[i]) ||         \
          -1e-5 > (a[i] - b[i])) {        \
        printf("Mismatch @ Iter %ld: calculated:" fmt " != expected:" fmt "\n", \
               i, a[i], b[i]);            \
        exit(1);                          \
      }                                   \
    }                                     \
  } while (0)

#endif // __TEST_H__

