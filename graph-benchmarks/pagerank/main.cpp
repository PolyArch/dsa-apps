#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "pr.dfg.h"
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <sstream>
#include <string>

#define NUM_THREADS 2
#define NUM_VERT_PER_THREAD V/NUM_THREADS
#define EFF_VERT_PER_THREAD (V/NUM_THREADS+1)

using namespace std;

// Barrier variable
pthread_barrier_t barr;
pthread_barrier_t barr2;
pthread_barrier_t barr3;

struct edge_info {
  uint16_t dst_id;
  // uint16_t wgt;
};

uint16_t cur_vertex_data[V+NUM_THREADS]; // output vector
uint16_t prev_vertex_data[V+NUM_THREADS]; // input vector
uint16_t offset[V+2*(NUM_THREADS-1)+1]; // matrix ptr
// Oh, this could be more than edges -- edges is without padding
// Also, I need to know the new value -- let's take it as a vector
// edge_info neighbor[E]; // matrix non-zero values
vector<uint16_t> neighbor; // matrix non-zero values

// implement the graph dataflow on a single core
// |___|___|
// |___|___|
// |___|___|
// Another difference is that it is dense vector, decide dataset size
// Should we exploit the fact that there is no point of weights? TODO: we can
// just count that const here -- real data to be read is the earlier pr value,
// and dest vertex id
// this should work on columns in range of tid, and a tile of vector
void mv(long tid) {

  uint64_t mask=0;
  for(int i=0; i<NUM_THREADS; ++i) {
    if(i!=tid) { addDest(mask,i); }
  }
  
  // 0..1676 (both included), 1677..3353
  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
    
  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SS_ATOMIC_SCR_OP(P_pr_addr, P_pr_val, 0, offset[end_col]-offset[start_col], 0);  
 
  SS_DMA_READ(&prev_vertex_data[start_col], 2, 2, end_col-start_col, P_pr_pass1);

  SS_VREPEAT_PORT(P_pr_row_size2);
  SS_RECURRENCE(P_pr_pass2, P_pr_prev_vert_pr, end_col-start_col);

  // TODO: add later
  // SS_VREPEAT_PORT(P_pr_row_size3);
  // SS_CONST(P_pr_R, 0, V);

  // last should be V+1-V, ... V-V-1,...1..0
  SS_CONST(P_pr_offset_list0,offset[start_col],1);

  SS_ADD_PORT(P_pr_offset_list0);
  SS_DMA_READ(&offset[start_col+1], 2, 2, end_col-start_col-1, P_pr_offset_list1);

  SS_CONST(P_pr_offset_list1,offset[end_col],1);

  // edge weight can be calculated inside dfg
  SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
  SS_INDIRECT_2D(P_pr_start_ind, &neighbor[0], end_col-start_col, 2, 2, P_pr_row_size1, P_pr_dest_id);

  uint16_t x;
  SS_RECV(P_pr_done, x);
  SS_RESET();

  SS_GLOBAL_WAIT();
  SS_WAIT_ALL();

}

void read_input_file() {
  string str(csr_file);
  FILE* graph_file = fopen(str.c_str(), "r");

  char linetoread[5000];

  cout << "start reading graph input file!\n";

  offset[0]=0;
  int prev_offset=0;
  int e=-1, prev_v=-1; // indices start from 0
  int prev_col_size=-1; int pad_size=-1;
  int part=0;
  bool pad_phase=false;
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    uint16_t src, dst, wgt;
    iss >> src >> dst; 
    ++e;
    
    if(src!=0 && pad_phase && src > 10) { // padded value: after a partition
      ++part;
      offset[prev_v+1+part] = e;
      prev_v = src;
      pad_phase = false;
    } else if(src!=prev_v && !pad_phase) {

      // Padding here
      if(prev_v==-1) {
        prev_col_size=0;
      } else {
        prev_col_size = e - offset[prev_v+part];
      }

      offset[prev_v+1+part]=e;
      // cout << (prev_v+1+part) << " OFFSET: " << e << endl;
      int k=prev_v+1+part;
      while(offset[--k]==0 && k>0) {
        offset[k]=prev_offset;
      }
      prev_offset=e;
      if(src==0 && prev_v!=-1) { // don't change at padding
        pad_phase=true;
      } else {
        prev_v = src;
      }
    }
    neighbor.push_back(dst);
  }
  // offset[V] = E;
  
  offset[V+2*(NUM_THREADS-1)] = neighbor.size();
  int k=V+2*(NUM_THREADS-1);
  while(offset[--k]==0 && k>0) { // offset[0] should be 0
    offset[k]=neighbor.size(); // prev_offset;
  }
  fclose(graph_file);
  cout << "Done reading graph file!\n";
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
  SS_CONFIG(pr_config, pr_size);
  mv(tid);
  end_roi();
  sb_stats();

  cout << "Returned back with tid: " << tid << endl;

  pthread_barrier_wait(&barr2);

  return NULL;
}
 
void init_prev_pr() {
  for(int i=0; i<V+NUM_THREADS; ++i) {
    prev_vertex_data[i] = 1;
  }
}

// FIXME: need to match for variable number of threads -- need better formula
void pad_both_at_end() {
  // add at the end of each partition
  // prev_vertex_data[V] = V; make sure it is V for NUM_THREADS=1
  for(int i=1; i<=NUM_THREADS; ++i) {
    // int ind = i*NUM_VERT_PER_THREAD;
    // prev_vertex_data[ind+i] = V;
    int ind = i*EFF_VERT_PER_THREAD;
    prev_vertex_data[ind-1] = V;
    cout << "Value entered at a location: " << ind-1 << endl; 
  }
}

void print_neighbor() {
  /*for(int i=0; i<=V+2; ++i) {
    cout << "Index pointer at column i: " << i << " is: " << offset[i] << endl;
  }*/
  cout << "Address at neighbor 0: " << &neighbor[0] << endl;
  for(unsigned i=0; i<neighbor.size(); ++i) {
    cout << "Neighbor at i: " << neighbor[i] << endl;
  }
}

int main() {
  read_input_file();
  init_prev_pr(); // make sure this is not 0 (actually I should fix my sentinal problem)
  pad_both_at_end();
  // print_neighbor();

  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  if(pthread_barrier_init(&barr2, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }


  if(pthread_barrier_init(&barr3, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }
 
  int final_num_threads = NUM_THREADS;
  pthread_t threads[final_num_threads];
  int rc;
  long t;
  for(t=0;t<final_num_threads;t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      return 0;
    }
  }

  for(int i = 0; i < final_num_threads; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return 0;
    }
  }
  return 0;
}

// different barrier
// SS_REM_PORT(P_pr_barrier_o, 1, mask, P_pr_barrier_i);
// cr_base_addr, stride, access_size, num_strides, val_por    t, scratch_type
/*
SS_CONST(P_pr_barrier_i, 1, 1);
SS_REM_SCRATCH(0, 8, 8, 1, P_pr_barrier_o, 0);
SS_WAIT_DF(NUM_THREADS-1,0);
*/
/*
SS_WAIT_ALL(); 
// FIXME: 2 problems here: 1) doesn't work -- packet to core 1 is lost (god knows may skip -- can see later -- 2 streams ne send kiya router ko to gadbad ho gayi jaane dete hn) 2) it's wrong
pthread_barrier_wait(&barr3);
int k=16; // should be 1
SS_CONST(P_pr_barrier_i, 1, k);
SS_REM_PORT(P_pr_barrier_o, k, mask, P_pr_barrier_i2);
// SS_REM_PORT(P_pr_barrier_o, 1, mask, P_pr_barrier_i2);
SS_SCR_WRITE(P_pr_barrier_o2, (NUM_THREADS-1)*8*k, 0); // this should be garbage though

SS_WAIT_ALL(); 
*/
