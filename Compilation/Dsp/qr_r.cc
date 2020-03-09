
#include <cmath>
#include <cstdio>
#include <iostream>

#include <sim_timing.h>

#ifndef N
#define N 32
#endif

#ifndef U1
#define U1 2
#endif

#ifndef U2
#define U2 2
#endif

double buffer[N * N];

void qr_r(double *a, double *q, double *tau) {
  double *w = buffer;
  double *v = buffer + N;
  w[0] = 1.0f;

  // #pragma ss config
  // {
  //   for (int i = 0; i < N - 1; ++i) {
  //     int n = N - i;
  //     double normx = 0;

  //     #pragma ss stream
  //     #pragma ss dfg dedicated unroll(U1)
  //     for (int j = i; j < N; ++j) {
  //       normx += a[i * N + j] * a[i * N + j];
  //     }

  //     double u1, out0, out1, temp = a[i * N + i];
  //     #pragma ss dfg temporal
  //     {
  //       double normx_ = sqrt(normx);
  //       double norm0 = 1. / sqrt(temp * temp);
  //       double s = -temp * norm0;
  //       out0 = s * normx_;
  //       u1 = 1.0f / (temp - s * normx_);
  //       out1 = s / u1 / normx_;
  //     }

  //     a[i * N + i] = out0;
  //     tau[i] = out1;

  //     #pragma ss stream
  //     #pragma ss dfg dedicated unroll(U1)
  //     for (int j = i + 1; j < N; ++j) {
  //       w[j - i] = a[i * N + j] * u1;
  //     }

  //     //householder done

  //     #pragma ss stream
  //     for (int k = 1; k < n; ++k) {
  //       double acc(0);

  //       #pragma ss dfg dedicated unroll(U1)
  //       for (int j = 0; j < n; ++j)
  //         acc += a[(k + i) * N + (j + i)] * w[j] * tau[i];

  //       #pragma ss dfg dedicated unroll(U1)
  //       for (int j = 1; j < n; ++j)
  //         a[(k + i) * N + (j + i)] -= tau[i] * w[j] * acc;
  //     }
  //     //std::cout << i << std::endl;
  //   }
  // }
  // //std::cout << "done!\n";

  #pragma ss config
  {
    for (int i = N - 2; i >= 0; --i) {
      int n = N - i;

      #pragma ss stream
      #pragma ss dfg datamove
      for (int j = i + 1; j < N; ++j)
        w[j - i] = a[j * N + i];

      #pragma ss stream
      for (int j = i; j < N; ++j) {
        double acc = 0.0;
        #pragma ss dfg dedicated unroll(U1)
        for (int k = i; k < N; ++k)
          acc += q[j * N + k] * w[k - i];
        #pragma ss dfg dedicated unroll(U2)
        for (int k = i; k < N; ++k)
          q[j * N + k] -= tau[i] * w[k - i] * acc;
      }
      //std::cout << i << std::endl;
    }
  }
}

double a[N * N], q[N * N], tau[N];

int main() {
  for (int i = 0; i < N * N; ++i)
    a[i] = i + 1;
  qr_r(a, q, tau);
  begin_roi();
  qr_r(a, q, tau);
  end_roi();
  sb_stats();
  return 0;
}
