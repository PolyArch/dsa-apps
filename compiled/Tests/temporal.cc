#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Common/test.h"

#define N 128

using namespace std;

void reference(double a, double b, double g) {
  {
    double c = a - b;
    double d = a + c;
    double e = a / c;
    double f = sqrt(d);
    double gg = e + f;
    assert(abs(gg - g) < 1e-5);
  }
}

int main() {

  double a, b, g;

  a = rand();
  b = rand();

  begin_roi();
  #pragma ss config
  {
    #pragma ss dfg temporal
    {
      double c = a - b;
      double d = a + c;
      double e = a / c;
      double f = sqrt(d);
      g = e + f;
    }
  }
  end_roi();
  sb_stats();

  reference(a, b, g);

  return 0;
}
