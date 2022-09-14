#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Common/multicore.h"
#include "../Specs/fft.h"

#define FFT_LOG2(x) ((x) ? ((31) - __builtin_clz((uint32_t)(x))) : 0)

void worka(int64_t span, TYPE real[N], TYPE img[N], TYPE real_twid[N], TYPE img_twid[N], int64_t cid) {
  #pragma ss config
  {
    int64_t outer_tc = N / span / 2;
    int64_t core_parts = NUM_CORES / outer_tc;
    int64_t chunk_no = cid / core_parts;
    int64_t span_start = (span / core_parts) * (cid % core_parts);
    // for (int64_t start = 0; start < N; start += span * 2) {
    int64_t start = chunk_no * span * 2;
    {
      #pragma ss stream
      #pragma ss dfg unroll(2)
      for (int64_t i = span_start; i < span_start + (span / core_parts); i++){
        int64_t even = i + start;
        int64_t odd = even + span;

        TYPE real_odd = real[odd];
        TYPE real_even = real[even];
        TYPE img_odd = img[odd];
        TYPE img_even = img[even];

        TYPE temp = real_even + real_odd;
        real_odd = real_even - real_odd;
        real_even = temp;

        temp = img_even + img_odd;
        img_odd = img_even - img_odd;
        img_even = temp;

        temp = real_twid[i] * real_odd - img_twid[i] * img_odd;
        img[odd] = real_twid[i] * img_odd + img_twid[i] * real_odd;
        real[odd] = temp;
        img[even] = img_even;
        real[even] = real_even;
      }
    }
    // }
  }
}

void workb(int64_t span, TYPE real[N], TYPE img[N], TYPE real_twid[N], TYPE img_twid[N], int64_t cid) {
  #pragma ss config
  {
    int64_t outer_tc = N / span / 2;
    int64_t outer_chunk = outer_tc / NUM_CORES;
    #pragma ss stream
    for (int64_t j = 0; j < outer_chunk; ++j) {
      int64_t start = (cid * outer_chunk + j) * span * 2;
      #pragma ss dfg unroll(2)
      for(int64_t i = 0; i < span; i++){
        int64_t even = i + start;
        int64_t odd = even + span;

        TYPE real_odd = real[odd];
        TYPE real_even = real[even];
        TYPE img_odd = img[odd];
        TYPE img_even = img[even];

        TYPE temp = real_even + real_odd;
        real_odd = real_even - real_odd;
        real_even = temp;

        temp = img_even + img_odd;
        img_odd = img_even - img_odd;
        img_even = temp;

        temp = real_twid[i] * real_odd - img_twid[i] * img_odd;
        img[odd] = real_twid[i] * img_odd + img_twid[i] * real_odd;
        real[odd] = temp;
        img[even] = img_even;
        real[even] = real_even;
      }
    }
  }
}

void work2(TYPE real[N], TYPE img[N], int64_t cid) {
  #pragma ss config
  {
    int64_t chunk = N / NUM_CORES;
    #pragma ss stream
    #pragma ss dfg unroll(2)
    for (int64_t i = 0; i < chunk; i += 4) {
      int64_t even0 = (cid * chunk) + i + 0;
      int64_t even1 = (cid * chunk) + i + 1;
      int64_t odd0  = (cid * chunk) + i + 2;
      int64_t odd1  = (cid * chunk) + i + 3;

      int64_t *ptr_real_odd01 = (int64_t*)&real[odd0];
      int64_t real_odd01 = *ptr_real_odd01;
      int64_t *ptr_real_even01 = (int64_t*)&real[even0];
      int64_t real_even01 = *ptr_real_even01;

      int64_t *ptr_img_odd01 = (int64_t*)&real[odd0];
      int64_t img_odd01 = *ptr_img_odd01;
      int64_t *ptr_img_even01 = (int64_t*)&real[even0];
      int64_t img_even01 = *ptr_img_even01;

      int64_t temp = fadd32x2(real_even01, real_odd01);
      real_odd01 = fsub32x2(real_even01, real_odd01);
      real_even01 = temp;

      temp = fadd32x2(img_even01, img_odd01);
      img_odd01 = fsub32x2(img_even01, img_odd01);
      img_even01 = temp;

      temp = fsub32x2(fmul32x2(real_odd01, 123), fmul32x2(img_odd01, 456));

      *ptr_img_odd01 = fsub32x2(fmul32x2(img_odd01, 123), fmul32x2(real_odd01, 456));
      *ptr_real_odd01 = temp;

      *ptr_img_even01 = img_even01;
      *ptr_real_even01 = real_even01;
    }
  }
}

void work1(TYPE real[N], TYPE img[N], int64_t cid) {
  #pragma ss config
  {
    int64_t chunk = N / NUM_CORES;
    #pragma ss stream
    #pragma ss dfg unroll(2)
    for (int64_t i = 0; i < chunk; i += 2) {

      int64_t *real64 = ((int64_t*)&real[i + cid * chunk]);
      int64_t *img64 = ((int64_t*)&img[i + cid * chunk]);

      int64_t real_even = fadd32x2(*real64, 1ll);
      int64_t real_odd = fsub32x2(*real64, 1ll);

      int64_t img_even = fadd32x2(*img64, 1ll);
      int64_t img_odd = fsub32x2(*img64, 1ll);

      *real64 = concat32x2(real_even, real_odd);
      *img64 = concat32x2(img_even, img_odd);
    }
  }
}

void fft(TYPE real[N], TYPE img[N], TYPE real_twid[N/2], TYPE img_twid[N/2], int64_t cid){
  int64_t span, stride;
  stride = 1;

  int64_t twid_carry = 0;

  for(span = N >> 1; N / (span * 2) < NUM_CORES; span >>= 1){
    worka(span, real, img, real_twid + twid_carry, img_twid + twid_carry, cid);
    twid_carry += span;
    barrier(NUM_CORES);
  }

  for(; span >= 4; span >>= 1){
    workb(span, real, img, real_twid + twid_carry, img_twid + twid_carry, cid);
    twid_carry += span;
    barrier(NUM_CORES);
  }

  work2(real, img, cid);
  barrier(NUM_CORES);
  work1(real, img, cid);
  barrier(NUM_CORES);

}



TYPE a_real[N], a_imag[N], w_real[N * 2], w_imag[N * 2];

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  fft(a_real, a_imag, w_real, w_imag, cid);
  barrier(nc);
  end_roi();

#ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
#else
  exit(0);
#endif
}
