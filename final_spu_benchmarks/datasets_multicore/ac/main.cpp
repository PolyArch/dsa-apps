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
#include "ac.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>

using namespace std;

#define NUM_THREADS 4
#define NUM_INPUTS 1
#define fused_const (1 | (1 & 0xFFFFFFFF00000000) << 32)
#define SCRATCH_SIZE 16384 // FIXME: number of kB
	
// const for all input (all of size V)
struct node_prop1 {
  vector<uint32_t> nodeType;
  vector<uint32_t> c0;
  vector<uint32_t> c1;
};

struct node_prop2 {
  float vr; 
  uint32_t flag;
};

// Barrier variable
pthread_barrier_t barr;

int V;
vector<node_prop2> ac[NUM_INPUTS];
// vector<float> dr[NUM_INPUTS]; // for float to fixed point
// vector<uint32_t> ac_dr[NUM_INPUTS]; // for fixed to float point
vector<float> ac_dr[NUM_INPUTS]; // for fixed to float point
node_prop1 ac_prop;

// start height and end height for each thread id
int start_index[NUM_THREADS];
int end_index[NUM_THREADS]; // last one is copy nodes

vector<int> height_ptr; 
vector<int> shadow_ptr; 

int spad_buffer = SCRATCH_SIZE/2;
// int linear_part_offset = (1<<15)/3;

// utility functions -------------

int getLinearAddr(int offset) {
  int linear_scr_offset = 1 << 14; // log of scratch size
  return linear_scr_offset|offset;
}

// TODO: give the aligned answer
int getLinearOffset(int part_id, int n_part) {
  int x = SCRATCH_SIZE*part_id/n_part;
  return ((x/8)*8);
}

int getBankedOffset(int part_id, int n_part) {
  int x = SCRATCH_SIZE*part_id/n_part;
  return ((x/8)*8);
}





// -----------------


// Which core's spad?
// load ac_prop in linear scratchpads
void load_linear_scratch(long tid) {
  int n_times = height_ptr[end_index[tid]]-height_ptr[start_index[tid]];
  int start_id = height_ptr[start_index[tid]];
  /*
  if(tid==0) {
  cout << "n_times: " << n_times << "\n";
  }
  */
  SB_DMA_SCRATCH_LOAD(&ac_prop.nodeType[start_id], 0, 4*n_times, 1, getLinearAddr(0));
  SB_DMA_SCRATCH_LOAD(&ac_prop.c0[start_id], 0, 4*n_times, 1, getLinearAddr(getLinearOffset(1,3)));
  SB_DMA_SCRATCH_LOAD(&ac_prop.c1[start_id], 0, 4*n_times, 1, getLinearAddr(getLinearOffset(2,3)));
  SB_WAIT_SCR_WR();
  // SB_WAIT_ALL();
}

void compute(long tid, int input_id) {

  // cout << "Compute for input id: " << input_id << "\n";
  int vr_offset = getBankedOffset(input_id%2,2);
  SB_CONFIG(ac_config,ac_size);

  // for(int h=start_index[tid]; h<end_index[tid]; ++h) {
	// Should start from less than the last level (for only 0 level node -- no
	// shadow node)
  for(int h=start_index[tid]+1; h<end_index[tid]; ++h) {
	int n_times = height_ptr[h+1]-height_ptr[h];
	// padding for indirect ports
	n_times = (n_times/2)*2;
	// cout << "N_times: " << n_times << "\n";
	int start_id = height_ptr[h];
	int linear_offset = start_id-height_ptr[h];
    
	// reads from linear scratchpad
	SB_SCRATCH_READ(getLinearAddr(0+linear_offset), 4*n_times, P_ac_nodeType);
    SB_SCRATCH_READ(getLinearAddr(getLinearOffset(1,3)+linear_offset), 4*n_times, P_IND_1);
    SB_SCRATCH_READ(getLinearAddr(getLinearOffset(2,3)+linear_offset), 4*n_times, P_IND_2);

	// TODO: these child id's should be allotted to it's offset rather than
	// using it here (do during preprocessing)
	// c0-start_index[tid], c1-start_index[tid] (Okay, well if I use normal
	// output ports for this -- my padding issue would be resolved maybe)

	SB_CONST(P_ac_const, fused_const, n_times);

	// indirect reads from banked scratchpad
	// SB_CONFIG_INDIRECT1(T32, T32, sizeof(node_prop2), 2*sizeof(uint32_t));
	// _index_addr + index * _ind_mult + _offsets[_index_in_offsets]*_data_bytes;
	SB_CONFIG_INDIRECT1(T32, T32, 2, 2);
    SB_INDIRECT_SCR(P_IND_1, vr_offset-2*start_index[tid], 2*n_times, P_ac_c1vf);
    // SB_CONFIG_INDIRECT1(T32, T32, sizeof(node_prop2), 2*sizeof(uint32_t));
    // SB_CONFIG_INDIRECT1(T32, T32, sizeof(node_prop2), sizeof(uint32_t));
    SB_CONFIG_INDIRECT1(T32, T32, 2, 2);
    SB_INDIRECT_SCR(P_IND_2, (vr_offset+8*n_times-2*start_index[tid]), 2*n_times, P_ac_c2vf);

	SB_WAIT_SCR_RD();

    // direct banked scratch write in sequence
	SB_SCR_WRITE(P_ac_vrf, 4*n_times*2, vr_offset+start_id);
	// SB_SCR_WRITE(P_ac_vr, 4*n_times, vr_offset+start_id);
	// SB_SCR_WRITE(P_ac_flag, 4*n_times, vr_offset+start_id+4*n_times);

    SB_WAIT_ALL();
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

   begin_roi();
   // double buffering here
   load_linear_scratch(tid);
   for(int i=0; i<NUM_INPUTS; ++i){
     compute(tid, i);
   }
   end_roi();
   sb_stats();
   printf("Forward propagation done!\n");
   // pthread_exit(NULL);
   return NULL;
}
/*
void receive_data(long tid){
  if(tid==0){
    SB_WAIT_SCR_WR();
  } else {
    int num_nodes = shadow_ptr[tid]-shadow_ptr[tid-1];
    SB_WAIT_DF(num_nodes, 0); // we do not need range because it considers only remote writes
  }
}

void send_data(long tid, int input_id){
  int src_offset = getBankedOffset(input_id%2,2);
  int rem_offset = getBankedOffset((input_id+1)%2,2);
  if(tid==0) {
    // SB_DMA_SCRATCH_LOAD(&ac[input_id][0].vr, 4, 4, total_nodes, src_offset);
    SB_DMA_SCRATCH_LOAD(&ac[input_id][0].vr, 4, 4, height_ptr[1]-height_ptr[0], src_offset);
  } else {  // send to the next core
	// TODO: decide this pattern
	// TODO: NEED TO APPLY MASK FOR THE DEST NODE IN SCR ADDR HERE
    int total_nodes = shadow_ptr[tid]-shadow_ptr[tid-1]; // fix this
    SB_SCR_REM_SCR(src_offset, total_nodes, total_nodes, 1, rem_offset, 0);
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

   begin_roi();
   // double buffering here
   load_linear_scratch(tid);
   for(int i=0; i<NUM_INPUTS+1; ++i){
	 if(i<NUM_INPUTS) {
	   send_data(tid, i);
	 }
	 if(i > 0) {
       receive_data(tid);
       compute(tid, i-1);
	 }
   }
   end_roi();
   sb_stats();
   printf("Forward propagation done!\n");
   // pthread_exit(NULL);
   return NULL;
}
*/

int main() {

  char lineToRead[5000];

  // read height ptr
  FILE *hgt = fopen("datasets/final_index.data", "r");

  while(fgets(lineToRead, 5000, hgt) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
	height_ptr.push_back(x);
  }
  fclose(hgt); 
  cout << "DONE READING HEIGHT POINTER\n";
 
  
  // read copy nodes index
  FILE *shadow = fopen("datasets/final_shadow_index.data", "r");

  while(fgets(lineToRead, 5000, shadow) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
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
	// cout << start_index[a/2] << " " << end_index[a/2] << "\n";
	h++; a+=2;
  }

  cout << "DONE ASSIGNING START AND END INDEX\n";

  // read final circuit data
  FILE *ckt = fopen("datasets/final_circuit.data", "r");
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
	    ac[i].resize(V);
	    ac_dr[i].resize(V);
	  }
	  ac_prop.nodeType.resize(V);
	  ac_prop.c0.resize(V);
	  ac_prop.c1.resize(V);
	  continue;
	}

	// TODO: apply those rules here
	if(op=='l'){
	  // cout << "recognized a leaf node\n";
	  iss >> var >> valid;
	  for(int i=0 ; i < NUM_INPUTS; ++i){
		ac[i][cur_v].vr = var;
	    ac_dr[i][cur_v] = 0.0f;
	  }

	  // cout << (cur_v*NUM_THREADS)/V << endl;
	  ac_prop.c0[cur_v] = height_ptr[end_index[(cur_v*NUM_THREADS)/V]];
	  ac_prop.c1[cur_v] = height_ptr[end_index[(cur_v*NUM_THREADS)/V]];
	  // ac_prop.c0[cur_v] = -1;
	  // ac_prop.c1[cur_v] = -1;
	} else {
	  // iss >> ac_prop.c0[cur_v] >> ac_prop.c1[cur_v] >> valid;
	  iss >> x >> y >> valid;
	  ac_prop.c0[cur_v] = (uint32_t)x;
	  ac_prop.c1[cur_v] = (uint32_t)y;
	  for(int i=0 ; i < NUM_INPUTS; ++i){
		ac[i][cur_v].vr = 0.0f;
	    ac_dr[i][cur_v] = 0.0f;
	  }
	}
    cout << "Child1: " << ac_prop.c0[cur_v] << " child2: " << ac_prop.c1[cur_v] << "\n"; 
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

  return 0;
}
