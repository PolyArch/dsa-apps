/***
 * Currently: double buffer + direct load
 * Next: double buffer + indirect load
 * Maybe: triple buffer + direct load
***/
#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <vector>
#include <assert.h>
#include <cstring>
#include "ac.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include <inttypes.h>
#include <sstream>

using namespace std;

#define NUM_THREADS 1
#define NUM_INPUTS 1
#define fused_const (1 | (1 & 0xFFFFFFFF00000000) << 32)
	
uint32_t *nodeType;
uint32_t *c0;
uint32_t *c1;

struct node_prop2 {
  float vr; 
  uint32_t flag;
};

// Barrier variable
pthread_barrier_t barr;

uint64_t x[1000];

int V;
// vector<node_prop2> ac[NUM_INPUTS];
// node_prop2 *ac[NUM_INPUTS];
float *ac_vr[NUM_INPUTS];
uint32_t *ac_flag[NUM_INPUTS];
// vector<float> ac_dr[NUM_INPUTS]; // for fixed to float point
float *ac_dr[NUM_INPUTS]; // for fixed to float point
// node_prop1 ac_prop;

// start height and end height for each thread id
int start_index[NUM_THREADS];
int end_index[NUM_THREADS]; // last one is copy nodes

vector<int> height_ptr; 
vector<int> shadow_ptr; 

// load ac_prop in linear scratchpads
void load_linear_scratch(long tid) {
  int n_times = height_ptr[end_index[tid]]-height_ptr[start_index[tid]];
  int num_bytes_from_scratch = min(4*n_times, getLinearOffset(1,3));
  num_bytes_from_scratch = (num_bytes_from_scratch/8)*8;
  // n_times = num_bytes_from_scratch/4;
  n_times = num_bytes_from_scratch/4;
     
  int start_id = height_ptr[start_index[tid]];
  start_id = (start_id/2)*2;
  SS_DMA_SCRATCH_LOAD(&nodeType[start_id], 1, 1, 4*n_times, getLinearAddr(0));
  SS_DMA_SCRATCH_LOAD(&c0[start_id], 1, 1, 4*n_times, getLinearAddr(getLinearOffset(1,3)));
  SS_DMA_SCRATCH_LOAD(&c1[start_id], 1, 1, 4*n_times, getLinearAddr(getLinearOffset(2,3)));
  // SS_WAIT_SCR_WR();
  SS_WAIT_ALL();
}

void compute(bool enable, long tid, int input_id) {
  if(!enable) return;
  // cout << "Compute for input id: " << input_id << "\n";
  // int vr_offset = getBankedOffset((1-input_id)%3,3);
  int vr_offset = getBankedOffset(0,3);
  int num_bytes_from_mem=0, num_bytes_from_scratch=0;
  int start_id=0;

  // Should start from less than the last level (for only 0 level node -- no shadow node)
  for(int h=start_index[tid]+1; h<=end_index[tid]; ++h) {
	int num_elem = height_ptr[h+1]-height_ptr[h];

    if(num_elem < 1) return;

	int num_iters = num_elem/2; // padding for indirect ports
    num_iters = (num_iters/2)*2; // everything even will really help


    // if cannot fit, then do half the layer -- split in 128 partitions maybe

	int linear_offset = height_ptr[h]-height_ptr[start_index[tid]];
    linear_offset = linear_offset*4;
	// cout << "NUM_ITERS: " << num_iters << " linear_offset: " << linear_offset << "\n";
    start_id = height_ptr[h];
    start_id = (start_id/2)*2;
    SS_DMA_READ(&nodeType[start_id], 8, 8, num_iters, P_ac_nodeType);
    SS_DMA_READ(&c0[start_id], 8, 8, num_iters, P_IND_1);
    SS_DMA_READ(&c1[start_id], 8, 8, num_iters, P_IND_2);
    
    /*
    num_bytes_from_mem = linear_offset + 8*num_iters - getLinearOffset(1,3); 
    if(num_bytes_from_mem>0) {

      num_bytes_from_scratch = 8*num_iters - num_bytes_from_mem;
      if(num_bytes_from_scratch<0) {
        num_bytes_from_mem = 8*num_iters;
        assert(num_bytes_from_mem%8==0); // Yes, it is!
        start_id = height_ptr[h]+num_bytes_from_scratch/4;
        start_id = (start_id/2)*2;
        SS_DMA_READ(&nodeType[start_id], 8, 8, num_bytes_from_mem/8, P_ac_nodeType);
        SS_DMA_READ(&c0[start_id], 8, 8, num_bytes_from_mem/8, P_IND_1);
        SS_DMA_READ(&c1[start_id], 8, 8, num_bytes_from_mem/8, P_IND_2);

      } else {
        num_bytes_from_scratch = (num_bytes_from_scratch/8)*8;
        num_bytes_from_mem = 8*num_iters - num_bytes_from_scratch; // more from scratch

	    SS_SCRATCH_READ(getLinearAddr(0+linear_offset), num_bytes_from_scratch, P_ac_nodeType);
        SS_SCRATCH_READ(getLinearAddr(getLinearOffset(1,3)+linear_offset), num_bytes_from_scratch, P_IND_1);
        SS_SCRATCH_READ(getLinearAddr(getLinearOffset(2,3)+linear_offset), num_bytes_from_scratch, P_IND_2);

        assert(num_bytes_from_mem%8==0); // Yes, it is!
        start_id = height_ptr[h]+num_bytes_from_scratch/4;
        start_id = (start_id/2)*2;
        SS_DMA_READ(&nodeType[start_id], 8, 8, num_bytes_from_mem/8, P_ac_nodeType);
        SS_DMA_READ(&c0[start_id], 8, 8, num_bytes_from_mem/8, P_IND_1);
        SS_DMA_READ(&c1[start_id], 8, 8, num_bytes_from_mem/8, P_IND_2);
      }
    } else {
	  // reads from linear scratchpad
      // 8 * num_iters bytes
	  SS_SCRATCH_READ(getLinearAddr(0+linear_offset), 8*num_iters, P_ac_nodeType);
      SS_SCRATCH_READ(getLinearAddr(getLinearOffset(1,3)+linear_offset), 8*num_iters, P_IND_1);
      SS_SCRATCH_READ(getLinearAddr(getLinearOffset(2,3)+linear_offset), 8*num_iters, P_IND_2);
    }
    
    */

	SS_CONST(P_ac_const, fused_const, num_iters);

	// indirect reads from banked scratchpad
	// _index_addr + index * _ind_mult + _offsets[_index_in_offsets]*_data_bytes;
	SS_CONFIG_INDIRECT1(T32, T32, 2, 2);
    // 4 * 2 * 2 * num_iters bytes
    // SS_INDIRECT_SCR(P_IND_1, vr_offset-2*start_index[tid], 2*num_iters, P_ac_c1vf);
    SS_INDIRECT_SCR(P_IND_1, 0, 2*num_iters, P_ac_c1vf);
    SS_CONFIG_INDIRECT1(T32, T32, 2, 2);
    // SS_INDIRECT_SCR(P_IND_2, (vr_offset+16*num_iters-2*start_index[tid]), 2*num_iters, P_ac_c2vf);
    // SS_INDIRECT_SCR(P_IND_2, 16*num_iters, 2*num_iters, P_ac_c2vf);
    SS_INDIRECT_SCR(P_IND_2, 0, 2*num_iters, P_ac_c2vf);

    // direct banked scratch write in sequence
    // SS_STRIDE(8,8);
	// SS_SCR_WRITE(P_ac_vrf, 8*2*num_iters, vr_offset+32*num_iters);
	// SS_DMA_WRITE(P_ac_vrf, 8, 8, 2*num_iters, &x[0]); // vr_offset+32*num_iters);
	SS_DMA_WRITE(P_ac_vrf, 8, 8, 2*num_iters-8, &x[0]); // vr_offset+32*num_iters);

    SS_WAIT_ALL();
  }
}

void receive_data(bool enable, long tid){
  if(!enable) return;
  if(tid==0){
    SS_WAIT_SCR_WR();
  } else {
    int num_nodes = shadow_ptr[tid]-shadow_ptr[tid-1];
    SS_WAIT_DF(num_nodes, 0);
  }
}

void send_data(bool enable, long tid, int input_id){
  if(!enable) return;

  int src_offset = getBankedOffset((2-input_id+3*NUM_THREADS)%3,3);
  int rem_offset = getRemoteBankedOffset(tid+1,(3-input_id)%3,3);

  if(tid==NUM_THREADS-1) { // send to memory
    // TODO: check! (both vr and flag?)
    // SS_SCRATCH_DMA_STORE(src_offset, 4, 4, (shadow_ptr[tid]-shadow_ptr[tid-1])*2, &ac[input_id][0].vr);
    // FIXME: might not give correct results
    SS_SCRATCH_DMA_STORE(src_offset, 4, 4, (shadow_ptr[tid]-shadow_ptr[tid-1]), &ac_vr[input_id][0]);
    SS_SCRATCH_DMA_STORE(src_offset, 4, 4, (shadow_ptr[tid]-shadow_ptr[tid-1]), &ac_flag[input_id][0]);
  } else {  // send to the next core
    int total_nodes = shadow_ptr[tid]-shadow_ptr[tid-1]; 
    // FIXME: source offset would be different
    // cout << "SOURCE OFFSET: " << src_offset << " REMOTE OFFSET: " << rem_offset << " number of nodes: " << total_nodes << endl;
    SS_SCR_REM_SCR(src_offset, 4, 4, total_nodes, rem_offset, 0);
  }
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

  int recv_offset = 0;   
  SS_CONFIG(ac_config,ac_size);
  load_linear_scratch(tid);
 
  begin_roi();
 // triple buffering here -- tid=1 will start after first round
  
  // for(int i=0; i<NUM_INPUTS+2; ++i){
  // for(int i=0; i<NUM_INPUTS+2+3*(NUM_THREADS-1); ++i){
  for(int i=0; i<4; ++i){

    bool cond1 = (i >= 3*tid) && (i<NUM_INPUTS+3*tid);
    bool cond2 = (i > 3*tid) && (i<NUM_INPUTS+1+3*tid);
    bool cond3 = (i>1+3*tid) && (i<NUM_INPUTS+2+3*tid);

    // bool tid_enable = (i >= tid*3);
    // bool cond1 = (i<NUM_INPUTS) && tid_enable;
    // bool cond2 = (i > 0) && (i<NUM_INPUTS+1) && tid_enable;
    // bool cond3 = (i>1) && tid_enable;

    if(i%3==0) {
      if(tid==0 && cond1) {
        recv_offset = getBankedOffset((3-i)%3,3);
        SS_DMA_SCRATCH_LOAD(&ac_vr[i][0], 4, 4, shadow_ptr[tid], recv_offset);
        // SS_DMA_SCRATCH_LOAD(&ac[i][0].vr, 4, 4, shadow_ptr[tid], recv_offset);
      }
      receive_data(cond1, tid); // doesn't work for tid=1 at the first time
      compute(cond2, tid, i-3*tid);
      send_data(cond3, tid, i-3*tid);
    } else if(i%3==1) {
      if(tid==0 && cond1) {
        recv_offset = getBankedOffset((3-i)%3,3);
        // SS_DMA_SCRATCH_LOAD(&ac[i][0].vr, 4, 4, shadow_ptr[tid], recv_offset);
        SS_DMA_SCRATCH_LOAD(&ac_vr[i][0], 4, 4, shadow_ptr[tid], recv_offset);
      }
      receive_data(cond1, tid);
      compute(cond2, tid, i-1-3*tid);
      send_data(cond3, tid, i-1-3*tid);
    } else {
      if(tid==0 && cond1) {
        recv_offset = getBankedOffset((3-i)%3,3);
        SS_DMA_SCRATCH_LOAD(&ac_vr[i][0], 4, 4, shadow_ptr[tid], recv_offset);
        // SS_DMA_SCRATCH_LOAD(&ac[i][0].vr, 4, 4, shadow_ptr[tid], recv_offset);
      }
      receive_data(cond1, tid);
      compute(cond2, tid, i-2-3*tid);
      send_data(cond3, tid, i-2-3*tid);
    }
    // do we need this?
    pthread_barrier_wait(&barr);
  }
  
  end_roi();
  sb_stats();
  printf("Forward propagation done!\n");
  // pthread_exit(NULL);
  return NULL;
}

int main() {

  char lineToRead[5000];

  // read height ptr
  // string str(index_file);
  string str(dataset);
  char a1[100] = "datasets/";
  char b1[100] = "/final_index.data";
  FILE *hgt = fopen(strcat(strcat(a1,str.c_str()),b1), "r");
  // FILE *hgt = fopen(str.c_str(), "r");
  // FILE *hgt = fopen("datasets/final_index.data", "r");

  while(fgets(lineToRead, 5000, hgt) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
    cout << x << " ";
	height_ptr.push_back(x);
  }
  fclose(hgt); 
  cout << "DONE READING HEIGHT POINTER\n";
 
  
  // read copy nodes index
  // str = shadow_file;
  char a2[100] = "datasets/";
  char b2[100] = "/final_shadow_index.data";
  FILE *shadow = fopen(strcat(strcat(a2,str.c_str()),b2), "r");
 
  // FILE *shadow = fopen(str.c_str(), "r");
  // FILE *shadow = fopen("datasets/final_shadow_index.data", "r");

  while(fgets(lineToRead, 5000, shadow) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
    cout << x << " ";
	shadow_ptr.push_back(x);
  }
  fclose(shadow); 
  cout << "DONE READING SHADOW NODE INDICES\n";

  // CHECKME: update start and end index (using shadow ptr)

  int a=0;
  for(unsigned h=0; h<height_ptr.size();){
	// cout << a << endl;
	start_index[a/2] = h;
	h++;
	while(h<height_ptr.size() && height_ptr[h]!=shadow_ptr[a]){
	  h++;
	}
	end_index[a/2] = h;
	cout << "SE: " << start_index[a/2] << " " << end_index[a/2] << "\n";
	h++; a+=2;
  }

  cout << "DONE ASSIGNING START AND END INDEX\n";

  // read final circuit data
  // str = circuit_file;
  char a3[100] = "datasets/";
  char b3[100] = "/final_circuit.data";
  FILE *ckt = fopen(strcat(strcat(a3,str.c_str()),b3), "r");
 
  // FILE *ckt = fopen(str.c_str(), "r");
  // FILE *ckt = fopen("datasets/final_circuit.data", "r");
  int cur_v=0;

  while(fgets(lineToRead, 5000, ckt) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    std::string level;
    char op;
	bool valid;
	float var;
	int x,y;

	iss >> op;

	if(op=='n'){
	  iss >> level >> V;
	  for(int i=0 ; i < NUM_INPUTS; ++i){
	    // ac[i].resize(V);
	    // ac_dr[i].resize(V);
        // ac[i] = (node_prop2*) aligned_alloc(64, V*sizeof(struct node_prop2));
        ac_vr[i] = (float*) aligned_alloc(8, V*sizeof(float));
        ac_flag[i] = (uint32_t*) aligned_alloc(8, V*sizeof(uint32_t));
        ac_dr[i] = (float*) aligned_alloc(8, V*sizeof(float));
	  }
	  // ac_prop.nodeType.resize(V);
	  // ac_prop.c0.resize(V);
	  // ac_prop.c1.resize(V);
      nodeType = (uint32_t*) aligned_alloc(8, V*sizeof(uint32_t));
      c0 = (uint32_t*) aligned_alloc(8, V*sizeof(uint32_t));
      c1 = (uint32_t*) aligned_alloc(8, V*sizeof(uint32_t));
	  continue;
	}

	// TODO: apply those rules here
	if(op=='l'){
	  // cout << "recognized a leaf node\n";
	  iss >> var >> valid;
	  for(int i=0 ; i < NUM_INPUTS; ++i){
		// ac[i][cur_v].vr = var;
		ac_vr[i][cur_v] = var;
	    ac_dr[i][cur_v] = 0.0f;
	  }

	  // cout << (cur_v*NUM_THREADS)/V << endl;
      // int range = end_index[(cur_v*NUM_THREADS)/V]-start_index[(cur_v*NUM_THREADS)/V];
      // cout << range << endl;
      // int x = rand()%range;
      // cout << x << endl;
      // x = x > 0 ? x : -x;
	  // c0[cur_v] = 0;
	  // c1[cur_v] = 0;
	
	  c0[cur_v] = height_ptr[end_index[(cur_v*NUM_THREADS)/V]];
	  c1[cur_v] = height_ptr[end_index[(cur_v*NUM_THREADS)/V]];
	  // ac_prop.c0[cur_v] = -1;
	  // ac_prop.c1[cur_v] = -1;
	} else {
	  // iss >> ac_prop.c0[cur_v] >> ac_prop.c1[cur_v] >> valid;
	  iss >> x >> y >> valid;
	  c0[cur_v] = (uint32_t)x;
	  c1[cur_v] = (uint32_t)y;
	  for(int i=0 ; i < NUM_INPUTS; ++i){
		// ac[i][cur_v].vr = 0.0f;
		ac_vr[i][cur_v] = 0.0f;
	    ac_dr[i][cur_v] = 0.0f;
	  }
	}
    // cout << "Child1: " << ac_prop.c0[cur_v] << " child2: " << ac_prop.c1[cur_v] << "\n"; 
	cur_v++;
  }
  fclose(ckt);  
  cout << "DONE READING CIRCUIT\n";

  /*
  start_index[0] = 0;
  for(unsigned i=0; i<shadow_ptr.size(); i+=2){
	end_index[i/2] = shadow_ptr[i];
	if(i>0){
	  start_index[i/2+1] = shadow_ptr[i+1];
	}
	cout << start_index[i/2] << " " << end_index[i/2] << "\n";
  }
*/
  SS_CONFIG(ac_config,ac_size);
  // load_linear_scratch(0);
  // compute(1,0,0);
  begin_roi();
  // compute(1,0,0);
  compute(1,1,0);
  end_roi();
  sb_stats();
  /*
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
  */

  return 0;
}

