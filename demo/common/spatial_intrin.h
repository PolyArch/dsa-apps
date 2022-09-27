#pragma once

// implement some wheels to unify c++ abi.

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifdef __clang__
#define HINTATTR __attribute__((optnone))
#else
#define HINTATTR
#endif


#ifdef __cplusplus
extern "C" {
#endif

static double fsqrt(double x) {
  return sqrt(x);
}

static double exp64(double x) {
  return exp(x);
}

static int64_t hladd64(int64_t x) {
  uint32_t full_mask = ~((uint32_t) 0);
  return (x & full_mask) + ((x >> 32) & full_mask);
}

static int64_t hladd32x2(int64_t x) {
  uint32_t full_mask = ~((uint32_t) 0);
  // return (x & full_mask) + ((x >> 32) & full_mask);
  return x + x;
}

static int64_t add16x4(int64_t a, int64_t b) {
  return a + b;
}

static int64_t madd64(int64_t a, int64_t b, int64_t c) {
  return a * b + c;
}

static int64_t mul16x4(int64_t a, int64_t b) {
  return a * b;
}

static int16_t div16(int16_t a, int16_t b) HINTATTR {
  return a / b;
}

static int64_t div16x4(int64_t a, int64_t b) HINTATTR {
  return a / b;
}

static int64_t fadd32x2(int64_t a, int64_t b) {
  return a + b;
}

static int64_t fsub32x2(int64_t a, int64_t b) {
  return a - b;
}

static int64_t fmul32x2(int64_t a, int64_t b) {
  return a * b;
}

static int64_t fhladd64(int64_t x) {
  return x + (x >> 32);
}

static int64_t concat64(int64_t a, int64_t b) {
  uint32_t full_mask = ~((uint32_t) 0);
  return (a & full_mask) | (b & full_mask) << 32;
}

static int64_t concat32(int64_t a, int64_t b) {
  uint32_t full_mask = ~((uint32_t) 0);
  return (a & full_mask) | (b & full_mask) << 32;
}

static int64_t concat32x2(int64_t a, int64_t b) {
  uint32_t full_mask = ~((uint32_t) 0);
  return (a & full_mask) | (b & full_mask) << 32;
}

static void arrayhint(const void *ptr, int64_t size, double reuse) HINTATTR {}

static void spadoverride(const void *ptr, int64_t addr, int64_t size) HINTATTR {}

static void* fifoize(void *ptr, int64_t n, int64_t cond, int64_t dtype,
                     int64_t if_sentinel, int64_t sentinel) HINTATTR {
  return (int8_t*)ptr + cond * dtype;
}


#define DECL_MAX_MIN(bits)                                                    \
  static int##bits##_t max##bits(int##bits##_t a, int##bits##_t b) HINTATTR { \
    return a > b ? a : b;                                                     \
  }                                                                           \
  static int##bits##_t min##bits(int##bits##_t a, int##bits##_t b) HINTATTR { \
    return a < b ? a : b;                                                     \
  }

DECL_MAX_MIN(64)
DECL_MAX_MIN(32)
DECL_MAX_MIN(16)
DECL_MAX_MIN(8)

static double fmax64(double a, double b) {
  return a > b ? a : b;
}

#ifdef __cplusplus
}
#endif

