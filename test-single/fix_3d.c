#include <cassert>
#include "testing.h"

uint64_t a[16], b[32];

int main() {
  /// 1, 2, 3, 4, ..., 16
  for (int i = 0; i < 16; ++i)
    a[i] = i + 1;

  SS_CONFIG(none_config, none_size);
  SS_DMA_RD_OUTER(16, 8, 0);
  SS_DMA_RD_OUTER(0, 2, 0);
  SS_DMA_RD_INNER(a, 16, P_none_in);
  SS_DMA_WRITE(P_none_out, 8, 8, 32, b);
  SS_WAIT_ALL();

  /// Expecting: 1, 2, 1, 2, 3, 4, 3, 4, ..., 15, 16, 15, 16
  for (int j = 0, start = 0; j < 32; j += 4, start += 2) {
    for (int i = 0; i < 2; ++i) {
      assert(b[j + i] == start + i + 1);
      assert(b[j + i + 2] == start + i + 1);
    }
  }
}
