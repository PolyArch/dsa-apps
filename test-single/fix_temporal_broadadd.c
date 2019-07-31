#include "testing.h"
#include "broad_temp.dfg.h"

uint64_t a[16];
uint64_t b[16];
uint64_t c[16];
uint64_t ref[16];


int main(int argc, char* argv[]) {
  for (int i = 0; i < 16; ++i) {
    a[i] = i + 1;
    b[i] = 16 - i;
    //ref[i] = 1 + 16;
  }

  begin_roi();
  SS_CONFIG(broad_temp_config, broad_temp_size);
  SS_FILL_MODE(STRIDE_ZERO_FILL);
  SS_CONFIG_PORT_EXPLICIT(16 * 4, -4);
  SS_DMA_READ(b, 8, 8, 16, P_broad_temp_A);
  SS_DMA_READ_STRETCH(b, 8, 128, -8, 16, P_broad_temp_B);
  SS_GARBAGE(P_broad_temp_C, 16 * 16 / 2 + 16);
  SS_WAIT_ALL();
  end_roi();

  //compare<uint64_t>(argv[0], c, ref,16);
}
