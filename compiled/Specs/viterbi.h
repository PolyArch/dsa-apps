#include <stdint.h>

#define TYPE double

typedef uint8_t tok_t;
typedef TYPE prob_t;
typedef uint8_t state_t;
typedef int64_t step_t;

//#define N_STATES 5
//#define N_OBS 32
//#define N_TOKENS 9
#define N_STATES  64
#define N_OBS     140
#define N_TOKENS  64
#define BATCH 16
