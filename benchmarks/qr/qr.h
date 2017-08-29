#ifndef gemm_h
#define gemm_h

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define eps 1e-4

#ifndef DTYPE
#define DTYPE float
#endif

void qr(DTYPE *, DTYPE *, DTYPE *);

#endif
