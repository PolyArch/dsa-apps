#include <complex>
#include "softbrain-config/fixed_point.h"
#include <cstdint>

void read_n_float_complex(FILE *file, int n, std::complex<float> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(file, "%f%f", &real, &imag);
    a[i] = complex<float>(real, imag);
  }
}

void read_n_fix_complex(FILE *file, int n, std::complex<int16_t> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(file, "%f%f", &real, &imag);
    a[i] = complex<float>(DOUBLE_TO_FIX(real), DOUBLE_TO_FIX(imag));
  }
}

bool compare_n_float_complex(FILE *ref_data, int n, complex<float> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag;
    fscanf(ref_data, "%f%f", &real, &imag);
    if (fabs(real - a[i].real()) + fabs(imag - a[i].imag()) > eps * 2) {
      printf("expected %f+%fi but %f+%fi\n", real, imag, a[i].real(), a[i].imag());
      return false;
    }
  }
  return true;
}

bool compare_n_fix_complex(FILE *ref_data, int n, complex<int16_t> *a) {
  for (int i = 0; i < n; ++i) {
    float real, imag, norm;
    fscanf(ref_data, "%f%f", &real, &imag);
    norm = real * real + imag * imag;
    if ((fabs(real - FIX_TO_DOUBLE(a[i].real())) + fabs(imag - FIX_TO_DOUBLE(a[i].imag()))) / norm  > eps) {
      printf("expect %f+%fi but %f+%fi\n", real, imag,
          FIX_TO_DOUBLE(a[i].real()), FIX_TO_DOUBLE(a[i].imag()));
      return false;
    }
  }
  return true;
}

