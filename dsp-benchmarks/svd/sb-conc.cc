#include "svd.h"
#include "matvec.h"
#include "sb_insts.h"
#include "aplygvs.dfg.h"
#include "fused.dfg.h"
#include "hhr.dfg.h"
#include "finalize.dfg.h"
#include "sim_timing.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_mul_cons(a, b) (a).real() * (b), (a).imag() * (b)

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

complex<float> f[_N_], d[_N_], r[_N_ * _N_], temp[_N_];
complex<float> _one(1, 0), _zero(0, 0);

void household(complex<float> *v, int n, complex<float> &alpha) {
  float norm0 = complex_norm(v[0]), norm1 = 0;
  for (int j = 1; j < n; ++j) {
    norm1 += complex_norm(v[j]);
  }
  float _alpha = sqrt(1 + norm1 / norm0);
  alpha = complex<float>(-v->real() * _alpha, -v->imag() * _alpha);
  float rate = 1 + _alpha;
  v[0] = complex<float>(v->real() * rate, v->imag() * rate);
  norm1 += complex_norm(*v);
  norm1 = 1. / sqrt(norm1);
  for (int j = 0; j < n; ++j) {
    v[j] = complex<float>(v[j].real() * norm1, v[j].imag() * norm1);
  }
}

void givens(complex<float> &a, complex<float> &b, complex<float> &alpha) {
  float norm0 = complex_norm(a), norm1 = complex_norm(b);
  float r = sqrt(norm1 + norm0);
  complex<float> c(std::conj(a) / r);
  complex<float> s(std::conj(b) / r);
  alpha = r;
  a = c;
  b = s;
}

/* L Givens Rotation
 * (c        s     ) (a)
 * (s.conj  -c.conj) (b)
 * */
#define apply_givens(c, s, a, b) \
  do { \
    complex<float> \
      resa((c) * (a) + (s) * (b)), \
      resb(std::conj(s) * (a) - std::conj(c) * (b)); \
    a = resa; \
    b = resb; \
  } while(false)

void implicit_kernel(complex<float> *d, complex<float> *f, complex<float> *v, int n) {
  float mu = complex_norm(d[n - 1]);
  complex<float> a(complex_norm(d[0]) - mu), b(complex_conj_mul(f[0], d[0])), alpha;
  complex<float> c, s, extra;
  givens(a, b, alpha);

  c = a; s = b;
  SB_CONFIG(aplygvs_config, aplygvs_size);
  SB_CONST(P_aplygvs_C, *((uint64_t*)&c), _N_);
  SB_CONST(P_aplygvs_S, *((uint64_t*)&s), _N_);
  SB_DMA_READ(v, 8, 8, _N_, P_aplygvs_A);
  SB_DMA_READ(v + _N_, 8, 8, _N_, P_aplygvs_B);
  SB_DMA_WRITE(P_aplygvs_O0, 8, 8, _N_, v);
  SB_DMA_WRITE(P_aplygvs_O1, 8, 8, _N_, v + _N_);
  //SB_WAIT_ALL();
  //for (int i = 0; i < _N_; ++i) {
  //  lmm2x2(m0, m1, v[i], v[i + _N_]);
  //}

  a = d[0] * c + f[0] * std::conj(s);
  f[0] = d[0] * s - f[0] * std::conj(c);
  b = d[1] * std::conj(s);
  d[1] *= -std::conj(c);

  givens(a, b, d[0]);
  apply_givens(a, b, f[0], d[1]);
  if (n != 2) {
    b *= f[1];
    f[1] *= -std::conj(a);
  }

  for (int i = 1; i < n - 1; ++i) {
    a = f[i - 1];
    givens(a, b, f[i - 1]);
    c = a; s = b;
    a = d[i];

    a = d[i] * c + f[i] * s;
    f[i] = d[i] * std::conj(s) - f[i] * std::conj(c);
    b = d[i + 1] * s;
    d[i + 1] *= -std::conj(c);

    SB_CONST(P_aplygvs_C, *((uint64_t*)&c), _N_);
    SB_CONST(P_aplygvs_S, *((uint64_t*)&s), _N_);
    SB_DMA_READ(v + i * _N_, 8, 8, _N_, P_aplygvs_A);
    SB_DMA_READ(v + i * _N_ + _N_, 8, 8, _N_, P_aplygvs_B);
    SB_DMA_WRITE(P_aplygvs_O0, 8, 8, _N_, v + i * _N_);
    SB_DMA_WRITE(P_aplygvs_O1, 8, 8, _N_, v + i * _N_ + _N_);
    //for (int j = 0; j < _N_; ++j) {
    //  lmm2x2(m0, m1, v[i * _N_ + j], v[(i + 1) * _N_ + j]);
    //}

    givens(a, b, d[i]);
    c = a; s = b;
    apply_givens(c, s, f[i], d[i + 1]);

    if (i != n - 2) {
      b = s * f[i + 1];
      f[i + 1] *= -std::conj(c);
    }
  }
  SB_WAIT_ALL();
}

//The most horrible kernel so far
void bidiagonalize(complex<float> *a, complex<float> *f, complex<float> *d, complex<float> *q) {
  int N = _N_;
  const int w_spad = 8;
  const int r_trans_spad = N * 8;
  const int horizon_vec  = 8192;
  const int q_spad = N * 8;

  SB_CONTEXT(2);
  SB_CONFIG(fused_config, fused_size);
  //SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);
  SB_CONST_SCR(0, *((uint64_t*)&_one), 1);

  SB_CONTEXT(1);
  SB_CONFIG(hhr_config, hhr_size);
  SB_CONST_SCR(0, *((uint64_t*)&_one), 1);
  //SB_DMA_SCRATCH_LOAD(&_one, 8, 8, 1, 0);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_A);
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_B);
  SB_CONST(P_hhr_Coef, 1065353216, N - 1);
  SB_CONST(P_hhr_reset, 2, N - 2);
  SB_CONST(P_hhr_reset, 1, 1);
  SB_REPEAT_PORT(3);
  SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
  SB_REPEAT_PORT(3);
  SB_DMA_READ(a, 8, 8, 1, P_hhr_HEAD);
  SB_CONST(P_hhr_Inst, 0, 1);
  SB_CONST(P_hhr_Inst, 1, 1);
  SB_CONST(P_hhr_Inst, 2, 1);

  SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, d); //alpha
  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1); //normalize
  SB_CONST(P_hhr_Coef, 1065353216, N - 1);
  SB_REPEAT_PORT(N * (N - 1));
  SB_RECURRENCE(P_hhr_RES, P_hhr_Coef, 1); //tau
  SB_DMA_READ(a + N, 8 * N, 8, N - 1, P_hhr_A);
  SB_CONST(P_hhr_reset, 1, N - 1);

  SB_SCR_WRITE(P_hhr_O, 8 * (N - 1), w_spad);
  SB_WAIT_SCR_WR();

  SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * N, N - 1, P_hhr_B);
  SB_2D_CONST(P_hhr_reset, 2, N - 1, 1, 1, N - 1);
  //SB_DMA_WRITE(P_hhr_O, 8, 8, N - 1, buffer + N); test pass!
  //SB_GARBAGE(P_hhr_O, N - 1); test pass!
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_hhr_A);
  }

  SB_REPEAT_PORT(N);
  SB_RECURRENCE(P_hhr_O, P_hhr_RA, N - 1);
  SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * N, N - 1, P_hhr_RB);
  SB_2D_CONST(P_hhr_RSignal, 1, 1, 0, N - 1, N - 1);
  //SB_DMA_WRITE(P_hhr_R_MEM, 8, 8, N - 1, a + 1);
  SB_SCR_WRITE(P_hhr_R_MEM, 8 * (N - 1), horizon_vec);
  SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (N - 1) * (N - 1), r_trans_spad);
  for (int i = 1; i < N; ++i) {
    SB_DMA_READ(a + i, 8 * N, 8, N, P_hhr_RC);
  }

  SB_WAIT_SCR_WR();

  SB_SCRATCH_READ(horizon_vec + 8, 8 * (N - 2), P_hhr_A);
  SB_SCRATCH_READ(horizon_vec + 8, 8 * (N - 2), P_hhr_B);
  SB_CONST(P_hhr_Coef, 1065353216, N - 2);
  SB_CONST(P_hhr_reset, 2, N - 3);
  SB_CONST(P_hhr_reset, 1, 1);
  SB_REPEAT_PORT(3);
  SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
  SB_REPEAT_PORT(3);
  SB_SCRATCH_READ(horizon_vec, 8, P_hhr_HEAD);
  SB_CONST(P_hhr_Inst, 0, 1);
  SB_CONST(P_hhr_Inst, 1, 1);
  SB_CONST(P_hhr_Inst, 2, 1);
  SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, f);
  SB_REPEAT_PORT(N - 2);
  SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1);
  SB_SCRATCH_READ(horizon_vec + 8, 8 * (N - 2), P_hhr_A);
  SB_CONST(P_hhr_Coef, 1065353216, N - 2);
  SB_CONST(P_hhr_reset, 1, N - 2);
  SB_RECURRENCE(P_hhr_O, P_hhr_IN, N - 2);
  SB_RECURRENCE(P_hhr_RES, P_hhr_IN, 1);
  SB_SCR_WRITE(P_hhr_OUTlocal, 8 * (N - 2), w_spad);
  //SB_DMA_WRITE(P_hhr_OUTremote, 0, 8 * (N - 2), 1, buffer); //DEBUG PASS!
  //SB_GARBAGE(P_hhr_OUTremote, N - 2); //TODO: xfer it to Q kernel later
  SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_IN, N - 2);
  SB_CONTEXT(2);
  SB_SCR_WRITE(P_fused_OUT, 8 * (N - 2), w_spad);
  //SB_DMA_SCRATCH_LOAD(&_one, 0, 8, 1, 0);
  SB_CONST_SCR(0, *((uint64_t*)&_one), 1);
  SB_CONTEXT(1);
  SB_REPEAT_PORT((N - 1) * (N - 1));
  SB_RECURRENCE(P_hhr_OUTlocal, P_hhr_Coef_, 1);
  SB_REPEAT_PORT(N - 1);
  SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_QTAU, 1);
  //SB_GARBAGE(P_hhr_OUTremote, 1); //TODO: xfer it to Q kernel later
  SB_WAIT_SCR_WR();
  SB_REPEAT_PORT(N - 1);
  SB_SCRATCH_READ(0, 8 * (N - 1), P_hhr_B_);
  SB_SCRATCH_READ(r_trans_spad, 8 * (N - 1) * (N - 1), P_hhr_A_);
  SB_CONST(P_hhr_ACC_, 0, N - 1);
  SB_RECURRENCE(P_hhr_O_, P_hhr_ACC_, (N - 1) * (N - 2));
  SB_SCR_WRITE(P_hhr_O_, 8 * (N - 1), horizon_vec);
  SB_WAIT_SCR_WR();
  //DEBUG INFO
  //SB_SCRATCH_DMA_STORE(0, 0, 8 * (N - 1), 1, hh);
  //SB_SCRATCH_DMA_STORE(horizon_vec, 0, 8 * (N - 1), 1, vec);
  //SB_SCRATCH_DMA_STORE(r_trans_spad, 0, 8 * (N - 1) * (N - 1), 1, r);
  SB_SCRATCH_READ(r_trans_spad, 8 * (N - 1) * (N - 1), P_hhr_RC);
  SB_SCR_PORT_STREAM(horizon_vec, 0, 8 * (N - 1), (N - 1), P_hhr_RB);
  SB_REPEAT_PORT(N - 1);
  SB_SCRATCH_READ(0, 8 * (N - 1), P_hhr_RA);
  SB_CONST(P_hhr_RSignal, 0, (N - 1) * (N - 1));
  SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (N - 1) * (N - 1), r_trans_spad);
  //SB_WAIT_SCR_WR();
  //DEBUG INFO
  //SB_SCRATCH_DMA_STORE(r_trans_spad, 0, 8 * (N - 1) * (N - 1), 1, r);

  SB_CONTEXT(2);
  SB_WAIT_SCR_WR();
  SB_SCR_PORT_STREAM(0, 8, 8, N - 1, P_fused_QW);
  SB_CONST(P_fused_QM, 1065353216, N - 1);
  SB_CONST(P_fused_Qreset, 1, N - 1);
  SB_REPEAT_PORT(N - 1);
  SB_RECURRENCE(P_fused_QV, P_fused_QA, N - 1);
  //SB_SCR_PORT_STREAM(0, 8, 8, N, P_fused_QA);
  SB_SCR_PORT_STREAM(0, 0, 8 * (N - 1), N - 1, P_fused_QB);
  SB_2D_CONST(P_fused_QC, 1065353216, 1, 0, N - 1, N - 2);
  SB_CONST(P_fused_QC, 1065353216, 1);
  SB_2D_CONST(P_fused_QSignal, 1, 1, 0, N - 2, N - 1);
  SB_DMA_WRITE(P_fused_Q_MEM, 0, 8 * (N - 1), 1, q + N + 1);
  SB_SCR_WRITE(P_fused_Q_SPAD, 8 * (N - 1) * (N - 2), q_spad);

  for (int i = 1; i < N - 1; ++i) {
    int n = N - i;
    SB_CONTEXT(1);
    SB_WAIT_SCR_WR();

    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_A);
    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_B);
    SB_CONST(P_hhr_Coef, 1065353216, n - 1);
    SB_CONST(P_hhr_reset, 2, n - 2);
    SB_CONST(P_hhr_reset, 1, 1);

    SB_REPEAT_PORT(3);
    SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
    SB_REPEAT_PORT(3);
    SB_SCRATCH_READ(r_trans_spad, 8, P_hhr_HEAD);
    SB_CONST(P_hhr_Inst, 0, 1);
    SB_CONST(P_hhr_Inst, 1, 1);
    SB_CONST(P_hhr_Inst, 2, 1);
    
    SB_SCRATCH_READ(r_trans_spad + 8, 8 * (n - 1), P_hhr_A);
    SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, d + i); //alpha
    SB_REPEAT_PORT(n - 1);
    SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1); //normalize
    SB_CONST(P_hhr_Coef, 1065353216, n - 1);
    SB_REPEAT_PORT(n * (n - 1));
    SB_RECURRENCE(P_hhr_RES, P_hhr_Coef, 1); //tau
    SB_CONST(P_hhr_reset, 1, n - 1);
    
    SB_SCR_WRITE(P_hhr_O, 8 * (n - 1), w_spad);
    SB_WAIT_SCR_WR();

    SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * n, n - 1, P_hhr_B);
    SB_2D_CONST(P_hhr_reset, 2, n - 1, 1, 1, n - 1);
    SB_SCRATCH_READ(r_trans_spad + 8 * n, 8 * (n - 1) * n, P_hhr_A);

    SB_REPEAT_PORT(n);
    SB_RECURRENCE(P_hhr_O, P_hhr_RA, n - 1);
    SB_SCR_PORT_STREAM(w_spad - 8, 0, 8 * n, n - 1, P_hhr_RB);
    SB_2D_CONST(P_hhr_RSignal, 1, 1, 0, n - 1, n - 1);
    SB_SCRATCH_READ(r_trans_spad + 8 * n, 8 * (n - 1) * n, P_hhr_RC);
    if (n - 1 != 1) {
      SB_SCR_WRITE(P_hhr_R_MEM, 8 * (n - 1), horizon_vec);
      SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (n - 1) * (n - 1), r_trans_spad);
    } else {
      SB_DMA_WRITE(P_hhr_R_MEM, 0, 8, 1, f + i);
      SB_DMA_WRITE(P_hhr_R_SPAD, 0, 8, 1, d + i + 1);
    }

    SB_WAIT_SCR_WR();

    //SB_SCRATCH_DMA_STORE(0, 0, 8 * n, 1, hh);
    //SB_WAIT_ALL();
    //for (int j = 0; j < n; ++j) std::cout << hh[j] << " "; std::cout << "\n";

    if (n > 2) {
      SB_SCRATCH_READ(horizon_vec + 8, 8 * (n - 2), P_hhr_A);
      SB_SCRATCH_READ(horizon_vec + 8, 8 * (n - 2), P_hhr_B);
      SB_CONST(P_hhr_Coef, 1065353216, n - 2);
      SB_CONST(P_hhr_reset, 2, n - 3);
      SB_CONST(P_hhr_reset, 1, 1);
      SB_REPEAT_PORT(3);
      SB_RECURRENCE(P_hhr_O, P_hhr_NORM, 1);
      SB_REPEAT_PORT(3);
      SB_SCRATCH_READ(horizon_vec, 8, P_hhr_HEAD);
      SB_CONST(P_hhr_Inst, 0, 1);
      SB_CONST(P_hhr_Inst, 1, 1);
      SB_CONST(P_hhr_Inst, 2, 1);
      SB_DMA_WRITE(P_hhr_RES, 8, 8, 1, f + i);
      SB_REPEAT_PORT(n - 2);
      SB_RECURRENCE(P_hhr_RES, P_hhr_B, 1);
      SB_SCRATCH_READ(horizon_vec + 8, 8 * (n - 2), P_hhr_A);
      SB_CONST(P_hhr_Coef, 1065353216, n - 2);
      SB_CONST(P_hhr_reset, 1, n - 2);
      SB_RECURRENCE(P_hhr_O, P_hhr_IN, n - 2);
      SB_RECURRENCE(P_hhr_RES, P_hhr_IN, 1);
      SB_SCR_WRITE(P_hhr_OUTlocal, 8 * (n - 2), w_spad);
      //SB_DMA_WRITE(P_hhr_OUTremote, 0, 8 * (n - 2), 1, hh); //DEBUG PASS!
      //SB_GARBAGE(P_hhr_OUTremote, n - 2); //TODO: xfer it to Q kernel later
      SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_IN, n - 2);
      SB_CONTEXT(2);
      SB_WAIT_SCR_WR();
      SB_SCR_WRITE(P_fused_OUT, 8 * (n - 2), w_spad);
      SB_CONTEXT(1);
      SB_REPEAT_PORT((n - 1) * (n - 1));
      SB_RECURRENCE(P_hhr_OUTlocal, P_hhr_Coef_, 1);
      SB_REPEAT_PORT((n - 1) * (N - 1));
      SB_XFER_RIGHT(P_hhr_OUTremote, P_fused_QTAU, 1);
      //SB_GARBAGE(P_hhr_OUTremote, 1); //TODO: xfer it to Q kernel later
      SB_WAIT_SCR_WR();
      SB_REPEAT_PORT(n - 1);
      SB_SCRATCH_READ(0, 8 * (n - 1), P_hhr_B_);
      SB_SCRATCH_READ(r_trans_spad, 8 * (n - 1) * (n - 1), P_hhr_A_);
      SB_CONST(P_hhr_ACC_, 0, n - 1);
      SB_RECURRENCE(P_hhr_O_, P_hhr_ACC_, (n - 1) * (n - 2));
      SB_SCR_WRITE(P_hhr_O_, 8 * (n - 1), horizon_vec);
      SB_WAIT_SCR_WR();
      //DEBUG INFO
      //SB_SCRATCH_DMA_STORE(0, 0, 8 * (n - 1), 1, hh);
      //SB_SCRATCH_DMA_STORE(horizon_vec, 0, 8 * (n - 1), 1, vec);
      //SB_SCRATCH_DMA_STORE(r_trans_spad, 0, 8 * (n - 1) * (n - 1), 1, r);
      SB_SCRATCH_READ(r_trans_spad, 8 * (n - 1) * (n - 1), P_hhr_RC);
      SB_SCR_PORT_STREAM(horizon_vec, 0, 8 * (n - 1), (n - 1), P_hhr_RB);
      SB_REPEAT_PORT(n - 1);
      SB_SCRATCH_READ(0, 8 * (n - 1), P_hhr_RA);
      SB_CONST(P_hhr_RSignal, 0, (n - 1) * (n - 1));
      SB_SCR_WRITE(P_hhr_R_SPAD, 8 * (n - 1) * (n - 1), r_trans_spad);
      //DEBUG INFO
      SB_WAIT_SCR_WR();

      /* It fixes the timing */
      //SB_SCRATCH_DMA_STORE(r_trans_spad, 0, 8 * (n - 1) * (n - 1), 1, r);
      SB_WAIT_ALL();
      //for (int j = 0; j < n - 1; ++j) std::cout << hh[j] << " "; std::cout << "\n";
      //std::cout << i << "\n";
      //std::cout << "r:\n";
      //for (int j = 0; j < n - 1; ++j) {
      //  for (int k = 0; k < n - 1; ++k)
      //    std::cout << r[k * (n - 1) + j] << " ";
      //  std::cout << "\n";
      //}
      /* It fixes the timing */

      SB_CONTEXT(2);
      SB_WAIT_SCR_WR();
      SB_SCR_PORT_STREAM(0, 0, 8 * (n - 1), N - 1, P_fused_QW);
      SB_SCRATCH_READ(q_spad, 8 * (N - 1) * (n - 1), P_fused_QM);
      SB_2D_CONST(P_fused_Qreset, 2, n - 2, 1, 1, N - 1);
      SB_REPEAT_PORT(n - 1);
      SB_RECURRENCE(P_fused_QV, P_fused_QA, N - 1);
      SB_SCR_PORT_STREAM(0, 0, 8 * (n - 1), N - 1, P_fused_QB);
      SB_SCRATCH_READ(q_spad, 8 * (N - 1) * (n - 1), P_fused_QC);
      SB_2D_CONST(P_fused_QSignal, 1, 1, 0, n - 2, N - 1);
      SB_DMA_WRITE(P_fused_Q_MEM, 0, 8 * (N - 1), 1, q + (i + 1) * N + 1);
      if (n - 2 != 1) {
        SB_SCR_WRITE(P_fused_Q_SPAD, 8 * (N - 1) * (n - 2), q_spad);
      } else {
        SB_DMA_WRITE(P_fused_Q_SPAD, 0, 8 * (N - 1), 1, q + (i + 2) * N + 1);
      }
    }
  }
  SB_CONTEXT(3);
  SB_WAIT_ALL();
  //for (int i = 0; i < N - 1; ++i) std::cout << f[i] << " "; std::cout << "\n";
  //for (int i = 0; i < N; ++i) std::cout << d[i] << " "; std::cout << "\n";
  //for (int i = 0; i < N; ++i) {
  //  for (int j = 0; j < N; ++j)
  //    std::cout << q[i * N + j] << " ";
  //  std::cout << "\n";
  //}
}

void svd(complex<float> *a, complex<float> *u, float *s, complex<float> *v) {
  bidiagonalize(a, f, d, v);

  int left = 0, right = _N_ - 1;
  while (left < right) {
    while (left < _N_ - 1 && fabs(f[left].real()) < eps && fabs(f[left].imag()) < eps)
      ++left;
    while (right >= 1 && fabs(f[right - 1].real()) < eps && fabs(f[right - 1].imag()) < eps)
      --right;
    if (right - left >= 1) {
      //std::cout << left << " " << right << "\n";
      implicit_kernel(d + left, f + left, v + left * _N_, right - left + 1);
      //for (int i = left; i < right; ++i) std::cout << f[i] << " "; std::cout << "\n";
      //for (int i = left; i < right + 1; ++i) std::cout << d[i] << " "; std::cout << "\n";
    }
  }

  int N = _N_;
  d[N - 1] = sqrt(complex_norm(d[N - 1]));
  SB_FILL_MODE(NO_FILL);
  SB_CONFIG(finalize_config, finalize_size);
  SB_DMA_READ(d, 0, 8 * _N_, 1, P_finalize_D);
  SB_SCR_WRITE(P_finalize_SINV, 8 * _N_, 0);
  SB_WAIT_SCR_WR();

  /*SB_DMA_READ(v, 0, 8 * _N_ * _N_, _N_, P_finalize_A);
  SB_2D_CONST(P_finalize_reset, 2, _N_ / 4 - 1, 1, 1, _N_ * _N_);
  //SB_RECURRENCE(P_finalize_O, P_finalize_U, _N_ * _N_);
  SB_REPEAT_PORT(_N_);
  SB_SCRATCH_READ(0, 8 * _N_, P_finalize_INV);
  SB_DMA_WRITE(P_finalize_RES, 0, 8 * _N_ * _N_, 1, u);
  for (int i = 0; i < _N_; ++i) {
    SB_DMA_READ(a + i * _N_, 0, 8 * _N_, _N_, P_finalize_B);
    s[i] = d[i].real();
  }
  SB_GARBAGE(P_finalize_O, _N_ * _N_);*/
  SB_WAIT_ALL();

  //for (int i = 0; i < _N_; ++i) {
  //  for (int j = 0; j < _N_; ++j) {
  //    complex<float> sum(0, 0);
  //    for (int k = 0; k < _N_; ++k)
  //      sum += complex<float>(complex_conj_mul(v[j * _N_ + k], a[i * _N_ + k]));
  //    u[i * _N_ + j] = sum;
  //  }
  //}
  /*for (int i = 0; i < _N_; ++i) {
    for (int j = 0; j < _N_; ++j) {
      u[i * _N_ + j] /= s[j];
    }
  }*/
}
