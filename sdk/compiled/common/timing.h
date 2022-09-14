#ifndef __TIMING_H__
#define __TIMING_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#ifdef __cplusplus
extern "C" {
#endif

// The DSA accelerator statistic can be collected
enum dsa_stat {
  // trigger and reset recording of statistic, return nothing
  BEGIN_ROI,
  END_ROI,
  CLEAR_ROI,
  // stream dispatcher statistic
  CTRL_LOAD_BITS,
  CTRL_CONF_NODES,
  CTRL_LAST_BITS,
  CTRL_EMPTY_QUEUE,
  CTRL_VALID_QUEUE,
  CTRL_FULL_QUEUE,
  // Processing Element Statistic
  cyc1stQuarterPeBusy,
  cyc2ndQuarterPeBusy,
  cyc3rdQuarterPeBusy,
  cyc4thQuarterPeBusy,
  cyc1stQuarterPeFired,
  cyc2ndQuarterPeFired,
  cyc3rdQuarterPeFired,
  cyc4thQuarterPeFired,
  // Switch Statistic
  cyc1stQuarterSwBusy,
  cyc2ndQuarterSwBusy,
  cyc3rdQuarterSwBusy,
  cyc4thQuarterSwBusy,
  cyc1stQuarterSwFired,
  cyc2ndQuarterSwFired,
  cyc3rdQuarterSwFired,
  cyc4thQuarterSwFired,
  // DMA Node Statistic
  CycDmaAlive,
  CycDmaNewStr,
  CycDmaAguReq,
  CycDmaReadPause,
  CycDmaWritePause,
  CycDmaPortPause,
  // Scratchpad Node Statistic
  CycSpmAlive,
  CycSpmNewStr,
  CycSpmAguReq,
  CycSpmReadPause,
  CycSpmWritePause,
  CycSpmPortPause,
  // Recurrence Node Statistic
  CycRecAlive,
  CycRecNewStr,
  CycRecAguReq,
  CycRecReadPause,
  CycRecWritePause,
  CycRecPortPause,
  // Generate Node Statistic
  CycGenAlive,
  CycGenNewStr,
  CycGenAguReq,
  CycGenReadPause,
  CycGenWritePause,
  CycGenPortPause,
  // Register Node Statistic
  CycRegAlive,
  CycRegNewStr,
  CycRegAguReq,
  CycRegReadPause,
  CycRegWritePause,
  CycRegPortPause,
  NUM_DSA_STAT
};

// Get statistic command
static __inline__ uint64_t get_stat(enum dsa_stat stat) {
  uint64_t statValue = 0;
  // 2 and 6 means a custom2 1 rd 1 rs RISC-V command
  // ROCC_INSTRUCTION_DS(2, statValue, stat, 6);
  return statValue;
}

// This is RISC-V specific way of getting current CPU cycle
#define rdcycle()                                  \
  ({                                               \
    unsigned long __tmp;                           \
    asm volatile("csrr %0, mcycle" : "=r"(__tmp)); \
    __tmp;                                         \
  })

static uint64_t ticks;

// Get the time of
static __inline__ uint64_t get_time(void) {
#ifdef CHIPYARD
  return rdcycle();
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
#endif
}

// Begin the recording of accelerator information
static void begin_roi() {
#ifdef GEM5
  asm volatile("add x0, x0, 1");
#else
#ifndef __x86_64
  asm volatile("fence");
#endif
  int64_t gc = 0;
  gc = get_stat(BEGIN_ROI);
  ticks = get_time();
#ifndef __x86_64
  asm volatile("fence");
#endif
#endif
}

// End the recording of accelerator information
static uint64_t end_roi() {
#ifdef GEM5
  asm volatile("add x0, x0, 2");
#else
#ifndef __x86_64
  asm volatile("fence");
#endif
  int64_t gc = 0;
  gc = get_stat(END_ROI);
  ticks = (get_time() - ticks);
  printf("roi ticks: %d\n", (int)ticks);
#ifndef __x86_64
  asm volatile("fence");
#endif
#endif
  return ticks;
}

// Clear the existing information of accelerator
static void clear_roi() {
  int64_t nouse = get_stat(CLEAR_ROI);
  return;
}

// Print out all dsa statistics
static void sb_stats() {
#ifdef GEM5
  asm volatile("add x0, x0, 3");
#endif
#ifdef CHIPYARD
  printf("dsa-stat-csv, ");
  for (int statIdx = CTRL_LOAD_BITS; statIdx < NUM_DSA_STAT; statIdx++) {
    int64_t stat = get_stat((enum dsa_stat) statIdx);
    printf("%d, ", (int)stat);
  }
  printf("\n");
#endif
}

#ifdef __cplusplus
}
#endif

#endif  // __TIMING_H__
