#include <complex>
#include <cstdio>
#include <cstdint>
#include <cuComplex.h>
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

void read_n_float_complex(FILE *file, int n, cuFloatComplex *a) {
  for (int i = 0; i < n; ++i){
    float real, imag;
    fscanf(file, "%f%f", &real, &imag);
    a[i] = make_cuFloatComplex(real/5.0,imag/5.0);
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

bool compare_n_float_complex(FILE *ref_data, int n, cuFloatComplex *a) {
  for (int i = 0; i <n ; ++i){
    float real, imag;
    fscanf(ref_data, "%f%f", &real, &imag);
    if (fabs(real - a[i].x) + fabs(imag - a[i].y) > eps * 2){
      printf("@%d expected %f+%fi but %f+%fi\n",i,real,imag,a[i].x,a[i].y);
      return false;
    }
  }
  return true;
}
