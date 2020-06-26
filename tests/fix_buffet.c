#include "testing.h"
#include <cassert>

int64_t a[16], b[32];

void shadow_11() {
  /// 1, 2, 3, 4, ..., 16
  for (int i = 0; i < 16; ++i)
    a[i] = i + 1;

  SS_CONFIG(none_config, none_size);
  SS_DMA_READ(a, 0, 8 * 16, 1, P_none_in);
  SS_BUFFET_ALLOCATE(0, 32, 8 * 16, P_none_out);
  SS_SCR_RD_OUTER(16, 8, 0);
  SS_SCR_RD_OUTER(0, 2, 0);
  SS_SCR_RD_INNER_BUFFET(16,
                         /*in=*/SCR_MEM_PORT, /*in_dtype*/T08,
                         /*shadow=*/MEM_SCR_PORT, /*shadow_dtype=*/T08);
  SS_2D_DCONST(MEM_SCR_PORT, 0, 31, 16, 1, 8, T08);
  SS_DMA_WRITE(SCR_MEM_PORT, 0, 32 * 8, 1, b);
  SS_WAIT_ALL();

  /// Expecting: 1, 2, 1, 2, 3, 4, 3, 4, ..., 15, 16, 15, 16
  for (int j = 0, start = 0; j < 32; j += 4, start += 2) {
    for (int i = 0; i < 2; ++i) {
      assert(b[j + i] == start + i + 1);
      assert(b[j + i + 2] == start + i + 1);
    }
  }
}

void shadow_24() {
  /// 1, 2, 3, 4, ..., 16
  for (int i = 0; i < 16; ++i)
    a[i] = i + 1;

  SS_CONFIG(none_config, none_size);
  SS_DMA_READ(a, 0, 8 * 16, 1, P_none_in);
  SS_BUFFET_ALLOCATE(0, 32, 8 * 16, P_none_out);
  SS_SCR_RD_OUTER(16, 8, 0);
  SS_SCR_RD_OUTER(0, 2, 0);
  SS_SCR_RD_INNER_BUFFET(16, /*in=*/SCR_MEM_PORT, /*in_dtype*/T16,
                         /*shadow=*/MEM_SCR_PORT, /*shadow_dtype=*/T32);
  SS_2D_DCONST(MEM_SCR_PORT, 0, 15, 16, 1, 8, T32);
  SS_DMA_WRITE(SCR_MEM_PORT, 0, 32 * 8, 1, b);
  SS_WAIT_ALL();

  /// Expecting: 1, 2, 1, 2, 3, 4, 3, 4, ..., 15, 16, 15, 16
  for (int j = 0, start = 0; j < 32; j += 4, start += 2) {
    for (int i = 0; i < 2; ++i) {
      assert(b[j + i] == start + i + 1);
      assert(b[j + i + 2] == start + i + 1);
    }
  }
}


int main() {
  shadow_11();
  shadow_24();
  return 0;
}
