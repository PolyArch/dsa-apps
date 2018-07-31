#include <complex>
#include "multi.dfg.h"
#include "sb_insts.h"

using std::complex;

#define N _N_

void solver(complex<float> *a, complex<float> *v) {
  SB_CONFIG(multi_config, multi_size);

  SB_DMA_READ(a, 8 * (N + 1), 8, N, P_multi_X);
  SB_CONST(P_multi_Y, *((uint64_t*) v), 1);

  SB_REPEAT_PORT((N - 2) / 4 + 1);
  SB_CONFIG_PORT_EXPLICIT((N - 1) * 2, -2);
  SB_RECURRENCE(P_multi_Y_X_REC, P_multi_V, N - 1);
  SB_GARBAGE(P_multi_Y_X_REC, 1);
  SB_DMA_WRITE(P_multi_Y_X_OUT, 8, 8, N, v);

  SB_FILL_MODE(STRIDE_ZERO_FILL);
  SB_DMA_READ_STRETCH(a + 1, 8 * (N + 1), 8 * (N - 1), -8, N - 1, P_multi_A);
  //SB_DMA_READ(v, 0, 8 * (N - 1), 1, P_multi_VV);
  SB_FILL_MODE(NO_FILL);

  int last_pad;
  {
    int pad = (4 - (N - 1) % 4) % 4;
    //SB_DMA_READ(a + 1, 0, 8 * (N - 1), 1, P_multi_A); SB_CONST(P_multi_A, 0, pad);
    SB_DMA_READ(v, 0, 8 * (N - 1), 1, P_multi_VV); SB_CONST(P_multi_VV, 0, pad);
    last_pad = pad;
  }

  for (int i = 1; i < N; ++i) {
    //int pad = (4 - (N - i - 1) % 4) % 4;
    int pad = (last_pad + 1) % 4;

    //SB_DMA_READ(a + i * N + i + 1, 0, 8 * (N - i - 1), 1, P_multi_A); SB_CONST(P_multi_A, 0, pad);

    //SB_REPEAT_PORT((N - i - 2) / 4 + 1);
    //if (i != N - 1) {
    //  SB_RECURRENCE(P_multi_Y_X_REC, P_multi_V, 1);
    //} else {
    //  SB_GARBAGE(P_multi_Y_X_REC, 1);
    //}

    SB_RECURRENCE(P_multi_O, P_multi_Y, 1);
    SB_RECURRENCE(P_multi_O, P_multi_VV, (N - i - 1));
    SB_CONST(P_multi_VV, 0, pad);
    SB_GARBAGE(P_multi_O, last_pad);

    last_pad = pad;
  }
  //SB_GARBAGE(P_multi_Y_X_REC, 1);
  SB_WAIT_ALL();
}

#undef N
