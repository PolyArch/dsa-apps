
#include "../Common/timing.h"
#include "../Common/test.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"

#ifndef N
#define N 48
#endif

#ifndef U1
#define U1 1
#endif

#ifndef U2
#define U2 1
#endif

#ifndef U3
#define U3 1
#endif

#ifndef U4
#define U4 1
#endif

double buffer[N * N];

void qr_q(double *a, double *q, double *tau) {
  double *w = buffer;

  #pragma ss config
  {
    double *spad = a;
    // double spad[N * N];
    double w[N];
    // #pragma ss stream
    // #pragma ss dfg dedicated
    // for (int i = 0; i < N * N; ++i) {
    //   spad[i] = a[i];
    // }
    // #pragma ss stream
    // #pragma ss dfg dedicated
    // for (int i = 0; i < N; ++i) {
    //   w[i] = a[i];
    // }
    for (int i = 0; i < N - 1; ++i) {
      int n = N - i;
      double normx = 0;

      #pragma ss stream nonblock
      #pragma ss dfg dedicated unroll(U1)
      for (int j = i; j < N; ++j) {
        normx += spad[i * N + j] * spad[i * N + j];
      }

      double u1, out0, out1, temp = spad[i * N + i];
      #pragma ss dfg temporal
      {
        double normx_ = fsqrt(normx);
        double norm0 = -1. / fsqrt(temp * temp);
        double s = temp * norm0;
        out0 = s * normx_;
        u1 = 1.0f / (temp - s * normx_);
        out1 = s / u1 / normx_;
      }

      a[i * N + i] = out0;
      tau[i] = out1;

      #pragma ss stream nonblock
      #pragma ss dfg dedicated unroll(U2)
      for (int j = i + 1; j < N; ++j) {
        w[j - i] = spad[i * N + j] * u1;
      }

      // //householder done
      #pragma ss stream
      for (int k = 1; k < n; ++k) {
        double acc = 0;

        #pragma ss dfg dedicated unroll(U3)
        for (int j = 0; j < n; ++j)
          acc += spad[(k + i) * N + (j + i)] * w[j];

        #pragma ss dfg dedicated unroll(U4)
        for (int j = 1; j < n; ++j)
          spad[(k + i) * N + (j + i)] -= tau[i] * w[j] * acc;
      }

    }
  }

  // #pragma ss config
  // {
  //   for (int i = N - 2; i >= 0; --i) {
  //     int n = N - i;
  //     #pragma ss stream
  //     #pragma ss dfg datamove
  //     for (int j = i + 1; j < N; ++j)
  //       w[j - i] = a[j * N + i];
  //     #pragma ss stream
  //     for (int j = i; j < N; ++j) {
  //       double acc = 0.0;
  //       #pragma ss dfg dedicated unroll(U2)
  //       for (int k = i; k < N; ++k)
  //         acc += q[j * N + k] * w[k - i];
  //       #pragma ss dfg dedicated unroll(U2)
  //       for (int k = i; k < N; ++k)
  //         q[j * N + k] -= tau[i] * w[k - i] * acc;
  //     }
  //   }
  // }
}

struct Arguments {
  double a[N * N], q[N * N], tau[N];
  double a_[N * N], q_[N * N], tau_[N];
} args_;

NO_SANITY_CHECK

NO_INIT_DATA

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    qr_q(args->a, args->q, args->tau);
  } else {
    qr_q(args->a_, args->q_, args->tau_);
  }
}
