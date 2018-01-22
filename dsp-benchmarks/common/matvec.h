#include <complex>
#include <algorithm>
#include "sb_insts.h"
#include "mmv.h"
#include "mmvc.h"
#include "vmm.h"
#include "vcmm.h"

#define complex_mul(a, b) (a).real() * (b).real() - (a).imag() * (b).imag(), \
  (a).real() * (b).imag() + (a).imag() * (b).real()

#define complex_mul_cons(a, b) (a).real() * (b), (a).imag() * (b)

#define complex_conj_mul(a, b) (a).real() * (b).real() + (a).imag() * (b).imag(), \
  (a).real() * (b).imag() - (a).imag() * (b).real()

#define complex_add(a, b) (a).real() + (b).real(), (a).imag() + (b).imag()

#define complex_sub(a, b) (a).real() - (b).real(), (a).imag() - (b).imag()

#define complex_norm(a) ((a).real() * (a).real() + (a).imag() * (a).imag())

#define get_pad(n, vec_width) (n % (vec_width) ? (vec_width) - (n % vec_width) : 0)

#define CPUvec_norm(from, ext, vec, res) \
  do { \
    res = 0; \
    for (int i(from); i < (from) + (ext); ++i) \
      res += complex_norm((vec)[i]); \
  } while(false) 

#define CPUvec_norm_and_copy(ext, vec, copy, res) \
  do { \
    res = 0; \
    for (int i = 0; i < (ext); ++i) { \
      res += complex_norm((vec)[i]); \
      (copy)[i] = (vec)[i]; \
    } \
  } while(false) 

#define CPUmat_mul_vec(upper, upper_ext, left, left_ext, mat_stride, mat, vec, is_conj, res) \
  do { \
    for (int i(upper); i < (upper) + (upper_ext); ++i) { \
      complex<float> sum(0, 0); \
      for (int j(left); j < (left) + (left_ext); ++j) \
        sum += (mat)[i * (mat_stride) + j] * ((is_conj) ? std::conj((vec)[j - (left)]) : (vec)[j - (left)]); \
      (res)[i - (upper)] = sum; \
    } \
  } while(false)

#define SBmat_mul_vec(upper, upper_ext, left, left_ext, mat_stride, mat, vec, is_conj, res) \
  do { \
    int pad(get_pad(left_ext, 4)); \
    int A(is_conj ? P_mmv_A : P_mmvc_A); \
    int B(is_conj ? P_mmv_B : P_mmvc_B); \
    int O(is_conj ? P_mmv_O : P_mmvc_O); \
    int reset(is_conj ? P_mmv_reset : P_mmvc_reset); \
    if (is_conj) {\
      SB_CONFIG(mmvc_config, mmvc_size); \
    } else { \
      SB_CONFIG(mmv_config, mmv_size); \
    } \
    for (int i(upper); i < (upper) + (upper_ext); ++i) { \
      SB_DMA_READ((mat) + i * (mat_stride), 8, 8, left_ext, A); \
      SB_CONST(A, 0, pad); \
      SB_DMA_READ((vec), 8, 8, (left_ext), B); \
      SB_CONST(B, 0, pad); \
      SB_CONST(reset, 0, ((left_ext) + pad) / 4 - 1); \
      SB_CONST(reset, 1, 1); \
      SB_GARBAGE(O, ((left_ext) + pad) / 4 - 1); \
      SB_DMA_WRITE(O, 8, 8, 1, (res) + i - (upper)); \
    } \
    SB_WAIT_ALL(); \
  } while(false)

#define REVELmat_mul_vec(upper, upper_ext, left, left_ext, mat_stride, mat, vec, is_conj, res) \
  do { \
    int pad(get_pad(left_ext, 4)); \
    int A(is_conj ? P_mmvc_A : P_mmv_A); \
    int B(is_conj ? P_mmvc_B : P_mmv_B); \
    int O(is_conj ? P_mmvc_O : P_mmv_O); \
    int reset(is_conj ? P_mmv_reset : P_mmvc_reset); \
    if (is_conj) {\
      SB_CONFIG(mmvc_config, mmvc_size); \
    } else { \
      SB_CONFIG(mmv_config, mmv_size); \
    } \
    SB_FILL_MODE(STRIDE_ZERO_FILL); \
    SB_DMA_READ((mat) + (upper) * (mat_stride), 8 * (left_ext), 8 * (mat_stride), upper_ext, A); \
    SB_DMA_READ((vec), 0, 8 * (left_ext), upper_ext, B); \
    SB_2D_CONST(reset, 2, ((left_ext) + pad) / 4 - 1, 1, 1, upper_ext); \
    SB_DMA_WRITE(O, 8, 8, upper_ext, res); \
    SB_WAIT_ALL(); \
  } while(false)

#define CPUvec_mul_mat(upper, upper_ext, left, left_ext, mat_stride, vec, is_conj, mat, res) \
  do { \
    for (int i(0); i < (left_ext); ++i) \
      (res)[i] = complex<float>(0, 0); \
    for (int j(upper); j < (upper) + (upper_ext); ++j) \
      for (int i(left); i < (left) + (left_ext); ++i) \
        (res)[i - (left)] += (mat)[j * (mat_stride) + i] * ((is_conj) ? std::conj((vec)[j - (upper)]) : (vec)[j - (upper)]); \
  } while (false)

#define SBvec_mul_mat(upper, upper_ext, left, left_ext, mat_stride, vec, is_conj, mat, res) \
  do { \
    int pad(get_pad((left_ext), 4)); \
    int A(is_conj ? P_vcmm_A : P_vmm_A); \
    int B(is_conj ? P_vcmm_B : P_vmm_B); \
    int C(is_conj ? P_vcmm_C : P_vmm_C); \
    int O(is_conj ? P_vcmm_O : P_vmm_O); \
    if (is_conj) { \
      SB_CONFIG(vcmm_config, vcmm_size); \
    } else { \
      SB_CONFIG(vmm_config, vmm_size); \
    } \
    SB_CONST(C, 0, (left_ext) + pad); \
    for (int i(upper); i < (upper) + (upper_ext); ++i) { \
      SB_DMA_READ((mat) + i * (mat_stride), 8, 8, (left_ext), A); SB_CONST(A, 0, pad); \
      SB_CONST(B, *((uint64_t*)(vec) + i - (upper)), ((left_ext) + pad) / 4); \
      if (i != (upper_ext) + (upper) - 1) { \
        SB_RECURRENCE(O, C, (left_ext) + pad); \
      } else { \
        SB_DMA_WRITE(O, 8, 8, (left_ext), (res)); \
        SB_GARBAGE(O, pad); \
      } \
    } \
    SB_WAIT_ALL(); \
  } while (false)


#define REVELvec_mul_mat(upper, upper_ext, left, left_ext, mat_stride, vec, is_conj, mat, res) \
  do { \
    int pad(get_pad(m, 4)); \
    int A(is_conj ? P_vcmm_A : P_vmm_A); \
    int B(is_conj ? P_vcmm_B : P_vmm_B); \
    int C(is_conj ? P_vcmm_C : P_vmm_C); \
    int O(is_conj ? P_vcmm_O : P_vmm_O); \
    if (is_conj) { \
      SB_CONFIG(vcmm_config, vcmm_size); \
    } else { \
      SB_CONFIG(vmm_config, vmm_size); \
    } \
    SB_FILL_MODE(STRIDE_ZERO_FILL); \
    SB_DMA_READ((mat) + (upper) * (mat_stride), 8 * (mat_stride), 8 * (left_ext), (upper_ext), A); \
    SB_CONST(C, 0, (left_ext) + pad); \
    SB_RECURRENCE(O, C, ((left_ext) + pad) * ((upper_ext) - 1)); \
    SB_REPEAT_PORT(((left_ext) + pad) / 4); \
    SB_DMA_READ((vec), 8, 8, (upper_ext), B); \
    SB_DMA_WRITE(O, 8, 8, (left_ext), (res)); \
    SB_GARBAGE(O, pad); \
    SB_WAIT_ALL(); \
  } while (false)
