// FIXME(@were): Not compiled.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#define N 128

struct Arguments {
  double a;
  double b;
  double g;
  double gg;
};

void run_reference(struct Arguments *args) {
  double a = args->a;
  double b = args->b;
  double g = args->g;
  {
    double c = a - b;
    double d = a + c;
    double e = a / c;
    double f = sqrt(d);
    double gg = e + f;
    args->gg = gg;
  }
}

int sanity_check(struct Arguments *args) {
  return args->gg - args->g < 1e-5 &&
    args->gg - args->g > -1e-5;
}

double temporal(double a, double b) {
  double g;
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
  return g;
}

void run_accelerator(struct Arguments *args, int x) {
  args->g = temporal(args->a, args->b);
}

