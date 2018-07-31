#include "svd.h"
#include "fileop.h"
#include <algorithm>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "sim_timing.h"
#include <cstring>

using std::complex;


int main() {

  run();
  begin_roi();
  run();
  end_roi();
  sb_stats();

  return 0;
}
