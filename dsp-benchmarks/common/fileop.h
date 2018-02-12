#include <complex>
#include <cstdio>
#include "ss-config/fixed_point.h"
#include <cstdint>

#ifndef eps
#define eps 1e-4
#endif

void read_n_float_complex(FILE *file, int n, std::complex<float> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(file, "%f%f", &real, &imag);
    a[i] = std::complex<float>(real, imag);
  }
}

void read_n_fix_complex(FILE *file, int n, std::complex<int16_t> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(file, "%f%f", &real, &imag);
    a[i] = std::complex<float>(DOUBLE_TO_FIX(real), DOUBLE_TO_FIX(imag));
  }
}

bool compare_n_float_complex(FILE *ref_data, int n, std::complex<float> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(ref_data, "%f%f", &real, &imag);
    if (fabs(real - a[i].real()) + fabs(imag - a[i].imag()) > eps * 2) {
      printf("@%d expected %f+%fi but %f+%fi\n", i, real, imag, a[i].real(), a[i].imag());
      return false;
    }
  }
  return true;
}

bool compare_n_fix_complex(FILE *ref_data, int n, std::complex<int16_t> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag, norm;
    fscanf(ref_data, "%f%f", &real, &imag);
    norm = real * real + imag * imag;
    if ((fabs(real - FIX_TO_DOUBLE(a[i].real())) + fabs(imag - FIX_TO_DOUBLE(a[i].imag()))) / norm  > eps) {
      printf("@%d expect %f+%fi but %f+%fi\n", i, real, imag,
          FIX_TO_DOUBLE(a[i].real()), FIX_TO_DOUBLE(a[i].imag()));
      return false;
    }
  }
  return true;
}

