#ifndef gemm_h
#define gemm_h

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define N 64
#define BN 4
#define eps 1e-3

#ifndef DTYPE
#define DTYPE float
#endif

void qr(DTYPE *, DTYPE *, DTYPE *);

#endif
