#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "matvec.h"
#include "sbdfg.h"

using std::complex;

void sqrt(complex<float> *a) {
  SB_CONFIG(sbdfg_config, sbdfg_size);
  for (int i = 0; i < N; i += 1) {
    SB_CONST(P_sbdfg_VAL, *((uint64_t*) a + i), 1);
    SB_RECV(P_sbdfg_O, a[i]);
    //a[i] = sqrt(a[i].real());
  }
}

