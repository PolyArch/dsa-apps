#include <stdint.h>
#include <stdio.h>

#ifdef __x86_64__

#include <sys/time.h>

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t ticks;
__attribute__ ((noinline))  void begin_roi() {
  ticks=rdtsc();
}
__attribute__ ((noinline))  void end_roi()   {
  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);
}

__attribute__ ((noinline)) static void sb_stats()   {
}
__attribute__ ((noinline)) static void sb_verify()   {
}

#else
static void begin_roi() {
    __asm__ __volatile__("add x0, x0, 1"); \
}
static void end_roi()   {
    __asm__ __volatile__("add x0, x0, 2"); \
}
static void sb_stats()   {
    __asm__ __volatile__("add x0, x0, 3"); \
}
static void sb_verify()   {
    __asm__ __volatile__("add x0, x0, 4"); \
}

#endif



//__attribute__ ((noinline)) static void begin_roi() {
//    __asm__ __volatile__("add x0, x0, 1"); \
//}
//__attribute__ ((noinline)) static void end_roi()   {
//    __asm__ __volatile__("add x0, x0, 2"); \
//}
//__attribute__ ((noinline)) static void sb_stats()   {
//    __asm__ __volatile__("add x0, x0, 3"); \
//}
//__attribute__ ((noinline)) static void sb_verify()   {
//    __asm__ __volatile__("add x0, x0, 4"); \
//}
//
//
