#include <iostream>
#include <ss-intrin/ss_insts.h>
#include <sim_timing.h>
#include "dfg.dfg.h"

#define N 64
#define K 64
#define M 64

#define BK 16

uint64_t a[N * K], b[K * M], c[N * M];

void kernel() {

  SS_CONFIG(dfg_config, dfg_size);

  SS_DMA_READ(b, 0, 8 * K * M, 1, MEM_SCR_PORT);
  SS_BUFFET_ALLOCATE(0, 8 * BK * M * 2, 8 * K * M, MEM_SCR_PORT);
  SS_SCR_RD_OUTER(8 * BK * M, K / BK, 0);
  SS_SCR_RD_OUTER(0, N, 0);
  SS_SCR_RD_INNER_BUFFET(8 * BK * M, P_dfg_a, T64, SCR_MEM_PORT, T64);
  SS_2D_DCONST(SCR_MEM_PORT, 0, BK * M * N - 1, BK * M * 8, 1, K / BK, T64);

  for (int ko = 0; ko < K; ko += BK) {
    //SS_DMA_READ(b + ko * M, 0, 8 * BK * M, 1, MEM_SCR_PORT);
    //SS_SCR_WRITE(MEM_SCR_PORT, 8 * BK * M, 0);
    //SS_WAIT_SCR_WR();
    for (int i = 0; i < N; ++i) {
      SS_DMA_READ(c + i * M, 0, 8 * M, 1, P_dfg_c);
      //SS_SCR_PORT_STREAM(0, 0, 8 * BK * M, 1, P_dfg_a);
      SS_REPEAT_PORT(M / 8);
      SS_DMA_READ(a + i * K, 0, 8 * BK, 1, P_dfg_b);
      SS_RECURRENCE(P_dfg_o, P_dfg_c, M * (BK - 1));
      SS_DMA_WRITE(P_dfg_o, 0, 8 * M, 1, c + i * M);
      //for (int ki = 0; ki < BK; ++ki) {
      //  for (int j = 0; j < M; ++j) {
      //    c[i * M + j] += a[i * K + k] * b[k * M + j];
      //  }
      //}
    }
  }
  SS_WAIT_ALL();

}

int main() {
  kernel();
  begin_roi();
  kernel();
  end_roi();
  sb_stats();
  return 0;
}
