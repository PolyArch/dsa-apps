#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <assert.h>
#include <pthread.h>
#include "eie.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include <inttypes.h>
#define NUM_THREADS	2

#define sentinal (SENTINAL16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48)


#define VTYPE uint16_t
using namespace std;

VTYPE *act_val;
VTYPE *act_ind;

int *wgt_row_ptr;
VTYPE *wgt_col_ind;
VTYPE *wgt_val;
uint64_t *out_vec;
int act_size;

// merged for padding: MERGING OF 4 ROWS
uint64_t *wgt_merged_col_ind;
uint64_t *wgt_merged_val;
int *wgt_merged_row_ptr;
uint64_t *act_merged_val;
uint64_t *act_merged_ind;

// Barrier variable
pthread_barrier_t barr;

// Each core will execute 1 merge (all activations read into 1st core from DRAM and then
// it broadcasts the output value; weights on the network from linear spad)
void mv_merged(long tid) {
  int ptr1=0, end1=0;
  int row_size=0;
  // std::cout << "reached till here\n";
  // load into scratchpads in main
 
  int ncol = N/NUM_THREADS; // N/63
  int start_col = ncol*tid;
  int end_col = start_col+ncol;
  // each thread id would have N/63 columns 
  // TODO: pad all of these (not just one) (--can i use fill mode here? No I guess)
  SB_CONFIG(eie_config,eie_size);

  int nweight_load = wgt_merged_row_ptr[end_col]-wgt_merged_row_ptr[start_col];
  cout << "Number of weights are: " << nweight_load << "\n";
  // since it is now using a port; either we should add a condition not to reset those ports
  SB_DMA_SCRATCH_LOAD(&wgt_merged_col_ind[wgt_merged_row_ptr[start_col]], 8, 8, nweight_load, (tid << 16) | 0);
  SB_DMA_SCRATCH_LOAD(&wgt_val[wgt_merged_row_ptr[start_col]], 8, 8, nweight_load, (tid << 16) | (4095)); // fix this offset
  SB_WAIT_SCR_WR();
 
  SB_DMA_WRITE(P_eie_out_val, 8, 8, ncol, &out_vec[0]); // write to the DMA? or are we doing multiple layers
  for (int i=tid; i<tid+ncol; ++i){ // work on a subset of columns
    ptr1 = wgt_merged_row_ptr[i];
    end1 = wgt_merged_row_ptr[i+1];
	if(end1-ptr1<=0)
	  continue;
	row_size = (end1-ptr1);
    
	SB_SCRATCH_READ((tid << 16) | 0, row_size*8, P_eie_wind);
	SB_CONST(P_eie_wind, sentinal, 1);
	SB_SCRATCH_READ((tid << 16) | 4095, row_size*8, P_eie_wval);
	SB_CONST(P_eie_wval, 0, 1);

	SB_REPEAT_PORT(4); // this might not be required after the port config
    SB_DMA_READ(&act_merged_ind[0], 8, 8, act_size, P_eie_aind); // being broadcasted from core 0
	SB_REPEAT_PORT(4);
    SB_DMA_READ(&act_merged_val[0], 8, 8, act_size, P_eie_aval); // being broadcasted from core 0
  }
  // error in this wait all
  SB_WAIT_ALL(); 
}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
     // exit(-1);
   }

   begin_roi();
   mv_merged(tid);
   end_roi();
   // pthread_exit(NULL);
   return NULL;
}

int main(){

  char lineToRead[5000];
  int nnz = (int)(M*act_sp);
  // make sure this is of the form 4k+3 (break after id is equal to that)
  int to_read_values = ((nnz-3)/4)*4+3;
  act_size = to_read_values+1;

  act_val = (VTYPE*)malloc(act_size*sizeof(VTYPE));
  act_ind = (VTYPE*)malloc(act_size*sizeof(VTYPE));
  int tind=0, tval=0;

  // READING ACTIVATIONS FIRST
  FILE *act_file = fopen("input_activations.data", "r");

  int id=0;
  // printf("Start reading activations file\n");

  while(fgets(lineToRead, 5000, act_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> tind >> tval;
	act_ind[id] = (VTYPE)tind;
	act_val[id] = (VTYPE)tval;
	if(id==(to_read_values-1)){
	  break;
	}
	id++;

  }

  act_ind[to_read_values] = SENTINAL16;
  act_val[to_read_values] = 0;
  // activations are being repeated so here, just push a sentinal at the end

  int id3=0;
  int merged_act_size = act_size/4;
  act_merged_ind = (uint64_t*)malloc(merged_act_size*sizeof(uint64_t));
  act_merged_val = (uint64_t*)malloc(merged_act_size*sizeof(uint64_t));

  VTYPE temp_val[4]; VTYPE temp_id[4];
  
  int count=0;
  for(int i=0; i<act_size; ++i){
	// temp_id[count] = act_val[i];
	// temp_val[count] = act_val[i];
    count++;

	if(count==4){
	  act_merged_ind[id3] = (act_ind[i] | act_ind[i-1] << 16 | (act_ind[i-2] & 0xFFFFFFFFFFFFFFFF) << 32 | (act_ind[i-3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  act_merged_val[id3] = (act_val[i] | act_val[i-1] << 16 | (act_val[i-2] & 0xFFFFFFFFFFFFFFFF) << 32 | (act_val[i-3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  id3++;
	  count=0;
	}
  }


  // printf("Done reading activations file\n");
  fclose(act_file);
  free(act_val);
  free(act_ind);

  // nnz = (int)(N*M*syn_sp);

  // 4N for extra padding
  wgt_val = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_col_ind = (VTYPE*)malloc((int)(N*M*syn_sp+4*N)*sizeof(VTYPE));
  wgt_row_ptr = (int*)malloc((N+1)*sizeof(int));

  int row_id; int prev_row_id=-1;
  int len=0;

  // READING WEIGHTS NOW
  FILE *weight_file = fopen("input_weights.data", "r");

  id=0;
  // printf("Start reading weights file\n");

  // Empty rows thing won't be a problem here
  while(fgets(lineToRead, 5000, weight_file) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> row_id >> tind >> tval;
	wgt_col_ind[id] = (VTYPE)tind;
	wgt_val[id] = (VTYPE)tval;

	if(row_id != prev_row_id){
	  if(prev_row_id!=-1)
	    len = id-wgt_row_ptr[prev_row_id];
	  else
		len = id;
	  
	  // std::cout << id-wgt_row_ptr[prev_row_id] << "\n";
	  wgt_row_ptr[row_id] = id;
	  prev_row_id = row_id;
	}
	id++;
  }
  // may check it!
  wgt_row_ptr[N] = id;
  
  // printf("Done reading weights file\n");
  
  fclose(weight_file);

  int syn_size = N*(M*syn_sp/4+1);
  wgt_merged_col_ind = (uint64_t*)malloc(syn_size*4*sizeof(uint64_t));
  wgt_merged_val = (uint64_t*)malloc(syn_size*4*sizeof(uint64_t));
  wgt_merged_row_ptr = (int*)malloc((int(N/4)+1)*4*sizeof(int));

  int id2=0;
  int counter=0;
  int offset=0;
  // printf("Entering the loop\n");
  for(int j=0; j<N/4; ++j){
	offset = j*4*M*syn_sp;
	// NEED TO MERGE 4 ROWS AT A TIME
    for(int i=0; i<(M*syn_sp); ++i){
	  wgt_merged_val[id2] = (wgt_val[offset+i] | wgt_val[offset+i+1] << 16 | (wgt_val[offset+i+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (wgt_val[offset+i+3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  wgt_merged_col_ind[id2] = (wgt_col_ind[offset+i] | wgt_col_ind[offset+i+1] << 16 | (wgt_col_ind[offset+i+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (wgt_col_ind[offset+i+3] & 0xFFFFFFFFFFFFFFFF) << 48);
	  id2++;
    }

    // wgt_merged_val[id2] = 0;
    // wgt_merged_col_ind[id2] = (SENTINAL16 | SENTINAL16 << 16 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32 | (SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48);
	// id2++;
	wgt_merged_row_ptr[j] = wgt_row_ptr[j*4]/4;
	// std::cout << "id2: " << id2 << "\n";
  }
  wgt_merged_row_ptr[N/4] = syn_size;
  free(wgt_val);
  free(wgt_col_ind);
  free(wgt_row_ptr);
  printf("Done writing merged weights values\n");
  /*
  for(int i=0; i<id2; ++i){
	std::cout << "ind: " << std::hex << wgt_merged_col_ind[i] << "\n";
	std::cout << "val: " << std::hex << wgt_merged_val[i] << "\n";
  }
  */
  out_vec = (uint64_t*)malloc(N/4*sizeof(uint64_t));
  // begin_roi();
  // mv_merged(wgt_merged_col_ind, wgt_merged_val, wgt_merged_row_ptr, act_merged_ind, act_merged_val, out_vec, merged_act_size);
  // end_roi();
  // sb_stats();
  assert(NUM_THREADS<C);
  
  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  int rc;
  long t;
  for(t=0;t<NUM_THREADS;t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);     
	if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
	  return 0;
    }
  }
  
  for(int i = 0; i < NUM_THREADS; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return -1;
    }
  }

  
  // begin_roi();
  // mv_mult(wgt_col_ind, wgt_val, wgt_row_ptr, act_ind, act_val, out_vec);
  // end_roi();
  
  return 0;
}
