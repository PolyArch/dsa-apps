#include "mkl_dfti.h"
#include <iostream>
#include <cstring>
#include "sim_timing.h"
#include "fileop.h"
#include <complex>

using std::complex;

complex<float> a[_N_], b[_N_];

int main() {
  FILE *input_data = fopen("input.data", "r"), *ref_data = fopen("ref.data", "r");
  if (!input_data || !ref_data) {
    puts("Data error!");
    return 1;
  }

  read_n_float_complex(input_data, _N_, a);
  memcpy(b, a, sizeof a);

  MKL_LONG status;

  DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
  DFTI_DESCRIPTOR_HANDLE my_desc2_handle;

  status = DftiCreateDescriptor( &my_desc1_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, _N_);
  status = DftiCommitDescriptor( my_desc1_handle );
  status = DftiComputeForward( my_desc1_handle, b);
  status = DftiFreeDescriptor(&my_desc1_handle);
  begin_roi();
  status = DftiCreateDescriptor( &my_desc1_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, _N_);
  status = DftiCommitDescriptor( my_desc1_handle );
  status = DftiComputeForward( my_desc1_handle, a);
  status = DftiFreeDescriptor(&my_desc1_handle);
  end_roi();

  return 0;
}
