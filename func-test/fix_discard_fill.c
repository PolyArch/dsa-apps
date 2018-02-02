#include "testing.h"
#include "add1_vec.h"

#define N 5
#define M 5
#define TOTAL ((N) * (M))

uint64_t input[TOTAL];
uint64_t output[TOTAL];
uint64_t answer[TOTAL];


int main(int argc, char* argv[]) {
  init();
  for (int i = 0; i < TOTAL; ++i) {
    input[i] = i;
    answer[i] = i + 1;
  }


  begin_roi();
  SB_CONFIG(add1_vec_config,add1_vec_size);
  SB_FILL_MODE(STRIDE_DISCARD_FILL);
  //SB_FILL_MODE(STRIDE_ZERO_FILL);

  SB_DMA_READ(input, M * 8, M * 8, N, P_add1_vec_in);
  SB_DMA_WRITE(P_add1_vec_out, M * 8, M * 8, N, output);
  //for (int i = 0; i < N; ++i) {
  //  SB_DMA_WRITE(P_add1_vec_out, 8, 8, M, output + i * M);
  //  SB_GARBAGE(P_add1_vec_out, 1);
  //}

  SB_WAIT_ALL();
  end_roi();

  compare<uint64_t>(argv[0], answer, output, TOTAL);
}
