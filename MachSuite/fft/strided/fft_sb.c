#include "fft.h"
#include "sb_insts.h"
#include "compute.h"

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
  int span, log, i;
  double temp;
  log = 0;

  SB_CONFIG(compute_config, compute_size);

  for(span=FFT_SIZE>>1; span; span>>=1, log++) {
    int times = FFT_SIZE / span / 2;

    SB_DMA_READ(real, 8 * span * 2, 8 * span, times, P_compute_EVE_REAL);
    SB_DMA_READ(img, 8 * span * 2, 8 * span, times, P_compute_EVE_IMAG);

    SB_DMA_READ(real + span, 8 * span * 2, 8 * span, times, P_compute_ODD_REAL);
    SB_DMA_READ(img + span, 8 * span * 2, 8 * span, times, P_compute_ODD_IMAG);

    for (i = 0; i < FFT_SIZE; i += span + span) {
      SB_DMA_READ(real_twid, (1 << log) * 8, 8, span, P_compute_TWD_REAL);
      SB_DMA_READ(img_twid, (1 << log) * 8, 8, span, P_compute_TWD_IMAG);

      /*for (even = i, odd = i + span; even < i + span; ++even, ++odd) {
        temp = real[even] + real[odd];
        real[odd] = real[even] - real[odd];
        real[even] = temp;

        temp = img[even] + img[odd];
        img[odd] = img[even] - img[odd];
        img[even] = temp;

        temp = real_twid[rootindex] * real[odd] - img_twid[rootindex] * img[odd];
        img[odd] = real_twid[rootindex]*img[odd] + img_twid[rootindex]*real[odd];
        real[odd] = temp;

        rootindex += 1 << log;
      }*/
    }
    SB_DMA_WRITE(P_compute_EVE_REAL_O, 8 * span * 2, 8 * span, times, real);
    SB_DMA_WRITE(P_compute_EVE_IMAG_O, 8 * span * 2, 8 * span, times, img);
    SB_DMA_WRITE(P_compute_ODD_REAL_O, 8 * span * 2, 8 * span, times, real + span);
    SB_DMA_WRITE(P_compute_ODD_IMAG_O, 8 * span * 2, 8 * span, times, img + span);

    SB_WAIT_ALL();
  }
}
