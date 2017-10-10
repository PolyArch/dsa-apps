#include "Dup32.h"
#include "../common/include/sb_insts.h"
#include "check.h"

int main() {
  int a[2] = {54321, 12345};
  SB_CONFIG(Dup32_config, Dup32_size);
  SB_DMA_READ(a, 8, 8, 1, P_Dup32_A);
  int high[2], low[2];
  int high_ans[] = {12345, 12345};
  int low_ans[] = {54321, 54321};
  SB_DMA_WRITE(P_Dup32_High, 8, 8, 1, high);
  SB_DMA_WRITE(P_Dup32_Low, 8, 8, 1, low);
  SB_WAIT_ALL();
  compare("dup_low",low, low_ans, 2);
  compare("dup_high", high, high_ans, 2);
  return 0;
}
