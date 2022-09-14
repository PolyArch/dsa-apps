// FIXME(@were): The join analysis is broken.

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/timing.h"

#define N 8

struct Arguments {
  uint64_t idx0[N / 2 + 1];
  uint64_t v0[N / 2 + 1];
  uint64_t idx1[N + 1];
  uint64_t v1[N + 1];
  uint64_t res;
} args_;

struct Arguments *init_data() {
#define INIT_WITH(attr, ...)                 \
  do {                                       \
    uint64_t values[] = {__VA_ARGS__};       \
    int n = sizeof(attr) / sizeof(uint64_t); \
    for (int i = 0; i < n; ++i) {            \
      attr[i] = values[i];                   \
    }                                        \
  } while (0)
  INIT_WITH(args_.idx0, 2, 3, 5, 11, 1ull << 63);
  INIT_WITH(args_.idx1, 1, 2, 3, 4, 5, 6, 7, 8, 1ull << 63);
  INIT_WITH(args_.v0, 1, 2, 3, 4, 0);
  INIT_WITH(args_.v1, 1, 2, 3, 4, 5, 6, 7, 8, 0);
#undef INIT_WITH
  return &args_;
}

void run_reference(struct Arguments *args) {
}

int join(uint64_t *idx0, uint64_t *idx1, uint64_t *v0, uint64_t *v1) {
  uint64_t res = 0;
  #pragma ss config
  {
    uint64_t acc = 0;
    #pragma ss stream fifo(v0,N/2)
    #pragma ss dfg
    for (int64_t i0 = 0, i1 = 0; i0 < N / 2 && i1 < N; ) {
      if (idx0[i0] == idx1[i1]) {
        acc += v0[i0] * v1[i1];
        ++i0;
        ++i1;
      } else {
        idx0[i0] < idx1[i1] ? ++i0 : ++i1;
      }
    }
    res = acc;
  }
  return res;
}

void run_accelerator(struct Arguments *args) {
  args->res = join(args->idx0, args->idx1, args->v0, args->v1);
}

int sanity_check(struct Arguments *args) {
  return args->res == 23;
}

