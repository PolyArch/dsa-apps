#include <complex>
#include "compute.dfg.h"
#include "sb_insts.h"

using std::complex;

#define N _N_

void solver(complex<float> *a, complex<float> *v) {
  SB_CONFIG(compute_config, compute_size);

  for (int i = 0; i < N; ++i) {
    v[i] /= a[i * N + i];

    int pad = (i + 1) % 4;
    pad = (4 - pad) % 4;

    SB_DMA_READ(a + i * N + N - i - 1, 8, 8, i + 1, P_compute_A);
    SB_CONST(P_compute_A, 0, pad);
    SB_DMA_READ(v + N - i - 1, 8, 8, i + 1, P_compute_VV);
    SB_CONST(P_compute_VV, 0, pad);

    SB_CONST(P_compute_V, *((uint64_t*) (v + i)), i / 4 + 1);

    SB_DMA_WRITE(P_compute_O, 8, 8, i + 1, v + N - i - 1);
    SB_GARBAGE(P_compute_O, pad);

    //for (int j = N - i - 1; j < N; ++j) {
    //  v[j] -= a[i * N + j] * v[i];
    //}
    SB_WAIT_ALL();
  }
}

#undef N
