#ifndef TESTING_H_
#define TESTING_H_

#include <algorithm>
#include <iostream>
#include <cassert>

template<typename T>
void init(T *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = (T)(i + 1);
}

template<typename T>
void init_rand(T *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = rand() % n;
}

template<typename T>
void comp(T *a, T *b, int n) {
  bool incorrect = false;
  for (int i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > 1e-5) {
      std::cout << i << ": " << a[i] << " " << b[i] << "\n";
      incorrect = true;
    }
  }
  assert(!incorrect);
}

template<typename T>
int cmp_int(T a, T b) {
  if (a > b)
    return 1;
  if (a < b)
    return -1;
  return 0;
}

#include <stdint.h>
#include <stdio.h>


#include <sys/time.h>

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t ticks;

static void begin_roi() {

#ifndef __x86_64__
  __asm__ __volatile__("add x0, x0, 1");
#else
  ticks=rdtsc();
#endif

}


static void end_roi()   {

#ifndef __x86_64__
  __asm__ __volatile__("add x0, x0, 2");
#else
  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);
#endif

}

static void sb_stats()   {
#ifndef __x86_64__
  __asm__ __volatile__("add x0, x0, 3");
#endif
}

static void sb_verify()   {
#ifndef __x86_64__
    __asm__ __volatile__("add x0, x0, 4");
#endif
}

#endif // TESTING_H
