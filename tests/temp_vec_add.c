#include "testing.h"
#include "temp_vec_add.dfg.h"

uint64_t a[ASIZE], b[ASIZE], c[ASIZE], ref[ASIZE];

void add(uint64_t *a, uint64_t *b, uint64_t *c) {
  SS_CONFIG(temp_vec_add_config, temp_vec_add_size);
  SS_DMA_READ(a, 0, ASIZE * 8, 1, P_temp_vec_add_A);
  SS_DMA_READ(b, 0, ASIZE * 8, 1, P_temp_vec_add_B);
  SS_DMA_WRITE(P_temp_vec_add_C, 0, ASIZE * 8, 1, c);
  SS_WAIT_ALL();
}

int main(int argc, char **argv) {
  for (int i = 0; i < ASIZE; ++i) {
    a[i] = i + 1;
    b[i] = ASIZE - i;
    ref[i] = ASIZE + 1;
  }

  add(a, b, c);
  compare<uint64_t>(argv[0], c, ref, ASIZE);
  begin_roi();
  add(a, b, c);
  end_roi();
  compare<uint64_t>(argv[0], c, ref, ASIZE);
  return 0;
}
