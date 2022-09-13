#include <stdio.h>

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/spatial_inrin.h"
#include "../Specs/fft.h"

#define FFT_LOG2(x) ((x) ? ((31) - __builtin_clz((uint32_t)(x))) : 0)

void work(int64_t span, int64_t start/*, volatile TYPE *_1, volatile TYPE *_2*/) {
  #pragma ss config
  {
    volatile TYPE real[N];
    volatile TYPE img[N];
    volatile TYPE real_twid[N];
    volatile TYPE img_twid[N];
    #pragma ss stream
    for (int64_t cstart = 0; cstart < (N / span / 2); ++cstart) {
      #pragma ss dfg unroll(2)
      for(int64_t i = 0; i < span; i++){
        int64_t start = cstart * span * 2;
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

        // int64_t rootindex = i * stride;

        TYPE real_w = real_twid[i];
        TYPE img_w = img_twid[i];
        temp = real_w * real_odd - img_w * img_odd;
        img[odd] = real_w * img_odd + img_w * real_odd;
        real[odd] = temp;
        img[even] = img_even;
        real[even] = real_even;
      }
    }
  }
}

void work2() {
  #pragma ss config
  {
    volatile TYPE real[N];
    volatile TYPE img[N];
    #pragma ss stream
    #pragma ss dfg unroll(1)
    for (int64_t i = 0; i < N; i += 4) {
      int64_t even0 = i + 0;
      int64_t even1 = i + 1;
      int64_t odd0  = i + 2;
      int64_t odd1  = i + 3;

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

void work1() {
  #pragma ss config
  {
    volatile TYPE real[N];
    volatile TYPE img[N];
    #pragma ss stream
    #pragma ss dfg unroll(2)
    for (int64_t i = 0; i < N; i += 2) {

      int64_t *real64 = ((int64_t*)&real[i]);
      int64_t *img64 = ((int64_t*)&img[i]);

      int64_t real_even = fadd32x2(*real64, 1ll);
      int64_t real_odd = fsub32x2(*real64, 1ll);

      int64_t img_even = fadd32x2(*img64, 1ll);
      int64_t img_odd = fsub32x2(*img64, 1ll);

      *real64 = concat32x2(real_even, real_odd);
      *img64 = concat32x2(img_even, img_odd);
    }
  }
}

void fft(TYPE real[N], TYPE img[N], TYPE real_twid[N*2], TYPE img_twid[N*2]){
  int64_t span, stride;
  stride = 1;

  int64_t twid_carry = 0;

  volatile TYPE ww_real[N];
  volatile TYPE ww_imag[N];
  volatile TYPE aa_real[N];
  volatile TYPE aa_imag[N];

  // #pragma ss config
  // {
  //   arrayhint(real, N * sizeof(TYPE), 1.0 - 1.0 / (FFT_LOG2(N)));
  //   arrayhint(img, N * sizeof(TYPE), 1.0 - 1.0 / (FFT_LOG2(N)));
  //   arrayhint(real_twid, N * sizeof(TYPE), 1.0 - 1.0 / (FFT_LOG2(N)));
  //   arrayhint(img_twid, N * sizeof(TYPE), 1.0 - 1.0 / (FFT_LOG2(N)));
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N; i += 2) {
  //     *((int64_t*)&aa_real[i]) = *((int64_t*)&real[i]);
  //   }
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N; i += 2) {
  //     *((int64_t*)&aa_imag[i]) = *((int64_t*)&img[i]);
  //   }
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N * 2; i += 2) {
  //     *((int64_t*)&ww_real[i]) = *((int64_t*)&real_twid[i]);
  //   }
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N * 2; i += 2) {
  //     *((int64_t*)&ww_imag[i]) = *((int64_t*)&img_twid[i]);
  //   }
  // }


  for(span = N >> 1; span >= 4; span >>= 1){
    work(span, twid_carry);
    twid_carry += span;
  }

  work2();
  work1();

  // #pragma ss config
  // {
  //   arrayhint(real, N * sizeof(TYPE), 11.0 / 12.0);
  //   arrayhint(img, N * sizeof(TYPE), 11.0 / 12.0);
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N; i += 2) {
  //     *((int64_t*)&real[i]) = *((int64_t*)&aa_real[i]);
  //   }
  //   #pragma ss stream
  //   #pragma ss dfg dedicated unroll(1)
  //   for (int64_t i = 0; i < N; i += 2) {
  //     *((int64_t*)&img[i]) = *((int64_t*)&aa_imag[i]);
  //   }
  // }
}



struct Arguments {
  TYPE a_real[N], a_imag[N], w_real[N * 2], w_imag[N * 2];
  TYPE a_real_[N], a_imag_[N], w_real_[N * 2], w_imag_[N * 2];
} args_;

struct Arguments *init_data() {
  return &args_;
}

void run_reference(struct Arguments *args) {
}

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    fft(args->a_real_, args->a_imag_, args->w_real_, args->w_imag_);
  } else {
    fft(args->a_real, args->a_imag, args->w_real, args->w_imag_);
  }
}

int sanity_check(struct Arguments *args) {
  return 1;
}
