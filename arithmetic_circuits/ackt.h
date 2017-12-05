#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
//#include "sim_timing.h"

#define eps 1e-4
//#define H 100
struct node {
  char nodeType;
  int index;
  double vr;
  double dr;
  bool flag;
  struct node *left;
  struct node *right;
};

void arith_ckt(struct node**, int, int *, int, double**, double**, bool**, bool**);

#endif
