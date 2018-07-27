#include "mkl_spblas.h"

#define _N_ N
#define _M_ M
#undef N
#undef M

#include "mkl.h"

#define N _N_
#define M _M_
#undef _N_
#undef _M_

#include "sim_timing.h"
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <vector>

using std::vector;


int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "[Usage] ./mkl.exe n m s0 s1\n");
    return 0;
  }
  int n = atoi(argv[1]);
  int m = atoi(argv[2]);
  float s0 = atof(argv[3]);
  float s1 = atof(argv[4]);

  float *vals;
  int*cols, *start, *end;

  sparse_matrix_t a;

  vector<float> _vals;
  vector<int> _cols;
  vector<int> _start;
  vector<int> _end;

  for (int i = 0; i < n; ++i) {
    bool started = false;
    for (int j = 0; j < m; ++j)
      if ((float) rand() / RAND_MAX < s0) {
        if (!started)
          _start.push_back(_vals.size());
        _vals.push_back((float) rand() / RAND_MAX);
        _cols.push_back(j);
        started = true;
    }
    _end.push_back(_vals.size());
  }

  assert(_start.size() == _end.size());
  assert(_vals.size() == _cols.size());


  assert(mkl_sparse_s_create_csr(&a, SPARSE_INDEX_BASE_ZERO, n, m, _start.data(), _end.data(), _cols.data(), _vals.data()) == SPARSE_STATUS_SUCCESS);

  float *xx = new float[m];
  float *yy = new float[n];

  float *x = new float[m];
  float *y = new float[n];

  matrix_descr md;
  md.type = SPARSE_MATRIX_TYPE_GENERAL;

  mkl_set_num_threads_local(NUM_OMP_THREADS);

  assert(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, md, xx, 0.0, yy) == SPARSE_STATUS_SUCCESS);

  begin_roi();
  mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, md, x, 0.0, y);
  end_roi();

  delete []x;
  delete []y;

  return 0;

}
