
#include "../Common/timing.h"
#include "../Common/test.h"
#include "../Common/spatial_inrin.h"
#include "../Common/interface.h"
#include "../Common/multicore.h"

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
double w[N];

void qr_q(double *a, double *q, double *tau, int64_t cid) {
  double *w = buffer;

  {
    double *spad = a;
    // double spad[N * N];
    // #pragma ss stream
    // #pragma ss dfg dedicated
    // for (int64_t i = 0; i < N * N; ++i) {
    //   spad[i] = a[i];
    // }
    for (int64_t i = 0; i < N - 1; ++i) {
      int64_t n = N - i;
      double normx = 0;

      if (cid == 0) {
        #pragma ss config
        {
          #pragma ss stream nonblock
          #pragma ss dfg dedicated unroll(1)
          for (int64_t j = i; j < N; ++j) {
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

          #pragma ss stream
          #pragma ss dfg dedicated unroll(1)
          for (int64_t j = i + 1; j < N; ++j) {
            w[j - i] = spad[i * N + j] * u1;
          }
        }
      }
      barrier(NUM_CORES);
      #pragma ss config
      {
        int64_t chunk = ((n - 2) / NUM_CORES) + 1;
        int64_t nn = (cid + 1) * chunk;
        nn = nn > n ? n : nn;
        // //householder done
        #pragma ss stream
        for (int64_t k = cid * chunk + 1; k < nn; ++k) {
          double acc = 0;
          #pragma ss dfg dedicated unroll(2)
          for (int64_t j = 0; j < n; ++j)
            acc += spad[(k + i) * N + (j + i)] * w[j];
          #pragma ss dfg dedicated unroll(2)
          for (int64_t j = 1; j < n; ++j)
            spad[(k + i) * N + (j + i)] -= tau[i] * w[j] * acc;
        }
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

double a[N * N], q[N * N], tau[N];

void thread_entry(int cid, int nc) {
  barrier(nc);
  begin_roi();
  qr_q(a, q, tau, cid);
  barrier(nc);
  end_roi();

  // return
#ifndef CHIPYARD
  sb_stats();
  if (cid != 0) {
    pthread_exit(NULL);
  }
#else
  exit(0);
#endif
}

