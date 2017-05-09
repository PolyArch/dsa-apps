#include "testing.h"

int main(int argc, char* argv[]) {
  begin_roi();
  SB_CONFIG(none_config,none_size);
  SB_WAIT_ALL();
  end_roi();
}
