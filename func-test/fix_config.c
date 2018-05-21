#include "testing.h"

#include "pred.dfg.h"

int main(int argc, char* argv[]) {
  begin_roi();
  SB_CONFIG(pred_config,pred_size);
  end_roi();
}
