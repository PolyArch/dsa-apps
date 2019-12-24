/*
* Assuming num of vert are divisible by num_threads
 * Assuming total edges per small partition is divisible by 4
 *
 *
*/

#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
// #include "pr.dfg.h"
#include "pr64.dfg.h"
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
// #include "m5op.h"
#include <inttypes.h>
#include <sstream>
#include <string>

// this should be strictly greater than C
#define NUM_THREADS 8
#define NUM_VERT_PER_THREAD (V/NUM_THREADS)
#define VEC_WIDTH 4

using namespace std;

// Barrier variable
pthread_barrier_t barr;

struct edge_info {
  uint64_t dst_id;
  // uint64_t wgt;
};

// #define V 3352 // 50516
// #define E 8859 // 1638612

uint64_t cur_vertex_data[V]; // output vector
uint64_t prev_vertex_data[V]; // input vector
uint64_t offset[V+1]; // matrix ptr
// Oh, this could be more than edges -- edges is without padding
// Also, I need to know the new value -- let's take it as a vector
// edge_info neighbor[E]; // matrix non-zero values
vector<uint64_t> neighbor; // matrix non-zero values

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

  begin_roi();
  // 0..1676 (both included), 1677..3353
  int start_col = tid*NUM_VERT_PER_THREAD;
  int end_col = (tid+1)*NUM_VERT_PER_THREAD; // not sure if correct
  // assert((offset[end_col]-offset[start_col])%4==0 && "number of edges belonging to a core should be divisible by vectorization width");
  // cout << "Elements corresponding to this: " << (offset[end_col]-offset[start_col]) << endl;

  // int E = offset[end_col] - offset[start_col];
  
  // okay, this is number of tuples created and sent to local/remote
  // SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  // atomic update of the page rank of vertices (location may be remote)

  // this is equal to the number of atomic update requests to be sent
  // Rest is satisfied by serving the bank queues...
  SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64);
  SS_ATOMIC_SCR_OP(P_pr64_addr, P_pr64_val, 0, offset[end_col]-offset[start_col], 0);  
 
  // SS_DMA_READ(&pr64ev_vertex_data[start_col], 2, 2, end_col-start_col, P_pr64_pass1);
  // read the page ranks of the active vertices in the previous iteration
  SS_DMA_READ(&prev_vertex_data[start_col], 8, 8, end_col-start_col, P_pr64_pass1);

  // It is not picking up from here
  // Why is repeat flag required where there is vrepeat port?
  // reuse times is data-dependent (source vertex is reused degree times for all its destination vertices)
  SS_VREPEAT_PORT(P_pr64_row_size2);
  SS_RECURRENCE(P_pr64_pass2, P_pr64_prev_vert_pr, end_col-start_col);

  // TODO: add later
  // SS_VREPEAT_PORT(P_pr_row_size3);
  // SS_CONST(P_pr_R, 0, V);

  // last should be V+1-V, ... V-V-1,...1..0
  
  // SS_DCONST(P_pr_offset_list0,offset[start_col],1, T16);
  // this is to calculate the degree (offset[start_col+1]-offset[start_col])
  SS_CONST(P_pr64_offset_list0,offset[start_col],1);
  SS_ADD_PORT(P_pr64_offset_list0); // while adding port, it gives 0
  SS_DMA_READ(&offset[start_col+1], 8, 8, end_col-start_col-1, P_pr64_offset_list1);
  SS_CONST(P_pr64_offset_list1,offset[end_col],1);
  // SS_DMA_READ(&offset[start_col+1], 2, 2, end_col-start_col-1, P_pr_offset_list1);


  // edge weight can be calculated inside dfg
  // SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
  // SS_INDIRECT_2D(P_pr64_start_ind, &neighbor[0], end_col-start_col, 2, 2, P_pr64_row_size1, P_pr64_dest_id);
  // accessing the neighbor array (initial index depends on the vertex)
  SS_CONFIG_INDIRECT(T64,T64,8); // multiplier for offset
  SS_INDIRECT_2D(P_pr64_start_ind, &neighbor[0], end_col-start_col, 8, 8, P_pr64_row_size1, P_pr64_dest_id);

  SS_GLOBAL_WAIT(NUM_THREADS);
 
  end_roi();
  sb_stats();


}

void read_input_file() {
  string str(csr_file);

  FILE* graph_file = fopen(str.c_str(), "r");

  char linetoread[5000];

  cout << "start reading graph input file!\n";

  offset[0]=0;
  int prev_offset=0;
  int e=-1, prev_v=0; // indices start from 1
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    int src, dst; //, wgt;
    // char ignore;
    iss >> src >> dst;// >> wgt;
    // cout << src << " " << dst << endl; // " " << wgt << endl;
    // cout << "prev_v: " << prev_v << endl;
    neighbor.push_back(dst);
    ++e;
    
    if(src!=prev_v) {
      offset[prev_v+1]=e;
      // cout << (prev_v+1) << " OFFSET: " << e << endl;
      int k=prev_v+1;
      while(offset[--k]==0 && k>0) {
        offset[k]=prev_offset;
        // cout << k << " OFFSET: " << prev_offset << endl;
      }
      prev_offset=e;
      prev_v=src;
      // cout << "index: " << (src) << " value: " << e << endl;
    }
    // cout << _neighbor[e].wgt << " " << _neighbor[e].dst_id << " " << _offset[prev_v-1] << endl;
  }
  offset[V] = E;
  prev_offset = E;
  int k=V;
  while(offset[--k]==0 && k>0) { // offset[0] should be 0
    offset[k]=prev_offset;
  }
  fclose(graph_file);
  cout << "Done reading graph file!\n";

  for(int i=0; i<V; ++i) {
    prev_vertex_data[i]=0;
  }
  cout << "Done initializing vertex data!\n";
}

void *entry_point(void *threadid) {

  long tid;
  tid = (long)threadid;
  
  SS_CONFIG(pr64_config, pr64_size);
  SS_GLOBAL_WAIT(NUM_THREADS);

  // Synchronization point
  int rc = pthread_barrier_wait(&barr);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
  {
    printf("Could not wait on barrier\n");
    // exit(-1);
  }

  mv(tid);
 
  // pthread_barrier_wait(&barr2);
  // cout << "Returned back with tid: " << tid << endl;

  return NULL;
}
 
void init_prev_pr() {
  for(int i=0; i<V; ++i) {
    prev_vertex_data[i] = 1;
  }
}

void fix_offset() {
  for(int i=0; i<NUM_THREADS; ++i) {
    int start_col = i*NUM_VERT_PER_THREAD;
    int end_col = (i+1)*NUM_VERT_PER_THREAD;

    cout << "start col: " << start_col << " end col: " << end_col << endl;
    // it says seg fault in accessing this?
    int modulo = (offset[end_col]-offset[start_col])%VEC_WIDTH;
    cout << "thread id: " << i << " start_offset: " << offset[start_col] << " end_offset: " << offset[end_col] << endl;
    if(modulo!=0) {
        // change offset
        int new_end = offset[end_col] - modulo;
        int k=end_col;
        while(k>=start_col && offset[k]>new_end) {
            cout << "col_id: " << k << " new end: " << new_end;
            offset[k]=new_end; --k;
        }
    }
    cout << "thread id: " << i << " start_offset: " << offset[start_col] << " end_offset: " << offset[end_col] << endl;
  }
}

int main() {
  read_input_file();
  init_prev_pr(); // make sure this is not 0 (actually I should fix my sentinal problem)
  /*for(int i=0; i<V+1; ++i) {
    cout << "Offset at i: " << i << " is: " << offset[i] << endl;
  }*/
  fix_offset();
  // m5_work_begin(0,0);
  // exit(0);
  // pad_both_at_end();
  // print_neighbor();

  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  /*if(pthread_barrier_init(&barr2, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }*/

/*
  if(pthread_barrier_init(&barr3, NULL, TOTAL_NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }*/
 
  int final_num_threads = NUM_THREADS;
  pthread_t threads[final_num_threads];
  int rc;
  long t;
  cout << "FINAL num threads: " << final_num_threads << endl;
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
  // m5_work_end(0,0);
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
