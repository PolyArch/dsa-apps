#pragma once

#define mc_min(a, b) ((a) < (b) ? (a) : (b))

#ifndef NUM_CORES
#define NUM_CORES 2
#endif

#ifndef CHIPYARD

#include <pthread.h>
#include <stdlib.h>

void barrier(int nc);

#else

void barrier(int nc);
void exit(int);

#endif
