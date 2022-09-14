#include <stdint.h>
#include "../Common/spatial_inrin.h"
#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Specs/viterbi.h"


struct Arguments {
  prob_t llike[N_OBS * N_STATES];
  tok_t obs[N_OBS];
  prob_t init[N_STATES];
  prob_t transition[N_STATES * N_STATES];
  prob_t emission[N_STATES * N_TOKENS];

  prob_t llike_[N_OBS * N_STATES];
  tok_t obs_[N_OBS];
  prob_t init_[N_STATES];
  prob_t transition_[N_STATES * N_STATES];
  prob_t emission_[N_STATES * N_TOKENS];
} args_;

void viterbi(tok_t * __restrict obs,
             prob_t *__restrict llike,
             prob_t *__restrict init,
             prob_t *__restrict transition,
             prob_t *__restrict emission) {

  prob_t min_p, p;
  state_t min_s, s;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
 
  // Initialize with first observation and initial probabilities
  // for( s=0; s<N_STATES; s++ ) {
  //   llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  // }

  #pragma ss config
  {
    for(int64_t t=1; t<N_OBS; t++ ) {
      #pragma ss stream nonblock
      for(int64_t curr=0; curr<N_STATES; curr++ ) {
        // prev = 0;
        min_p = 0; /*llike[t-1][prev] +
                transition[prev*N_STATES+curr] +
                emission[curr*N_TOKENS+obs[t]];*/
        #pragma ss dfg dedicated unroll(4)
        for(int64_t prev=1; prev<N_STATES; prev++ ) {
          p = llike[(t - 1) * N_STATES + prev] +
              transition[curr*N_STATES+prev] +
              emission[obs[t]*N_TOKENS+curr];
          min_p = fmax64(p, min_p);
        }
        llike[(t - 1) * N_STATES + curr] = min_p;
      }
    }
  }
}

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  if (_) {
    viterbi(args->obs_, args->llike_, args->init_, args->transition_, args->emission_);
  } else {
    viterbi(args->obs, args->llike, args->init, args->transition, args->emission);
  }
}


