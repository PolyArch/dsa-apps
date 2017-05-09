#include "testing.h"

int main(int argc, char* argv[]) {
  begin_roi();
  SB_WAIT_SCR_WR();
  end_roi();
}
