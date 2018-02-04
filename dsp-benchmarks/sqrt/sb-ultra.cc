#include <complex>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "matvec.h"
#include "another.dfg.h"

using std::complex;

void sqrt(complex<float> *a) {
  SB_CONFIG(another_config, another_size);
  for (int i = 0; i < N; i += 2) {
    SB_CONST(P_another_A, *((uint64_t*) a + i), 1);
    SB_RECV(P_another_OA, a[i]);
    SB_CONST(P_another_B, *((uint64_t*) a + i + 1), 1);
    SB_RECV(P_another_OB, a[i + 1]);
    //a[i] = sqrt(a[i].real());
  }
}

