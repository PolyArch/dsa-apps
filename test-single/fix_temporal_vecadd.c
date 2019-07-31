#include "testing.h"
#include "add_vec_temp.dfg.h"

uint64_t a[16];
uint64_t b[16];
uint64_t c[16];
uint64_t ref[16];


int main(int argc, char* argv[]) {
  for (int i = 0; i < 13; ++i) {
    a[i] = i + 1;
    b[i] = 16 - i;
    ref[i] = 1 + 16;
  }

  begin_roi();
  SS_CONFIG(add_vec_temp_config, add_vec_temp_size);
  SS_FILL_MODE(STRIDE_DISCARD_FILL);
  SS_DMA_READ(a, 8, 8, 16, P_add_vec_temp_A);
  SS_DMA_READ(b, 8, 8, 16, P_add_vec_temp_B);
  SS_DMA_WRITE(P_add_vec_temp_C, 8, 8, 16, c);

  SS_WAIT_ALL();
  end_roi();

  compare<uint64_t>(argv[0], c, ref,16);
}
