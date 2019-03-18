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

using namespace std;

// Barrier variable
pthread_barrier_t barr;
pthread_barrier_t barr2;

struct edge_info {
  uint16_t dst_id;
  // uint16_t wgt;
};

uint16_t cur_vertex_data[V+1]; // output vector
uint16_t prev_vertex_data[V+1]; // input vector
uint16_t offset[V+2]; // matrix ptr
// Oh, this could be more than edges -- edges is without padding
// Also, I need to know the new value -- let's take it as a vector
// edge_info neighbor[E]; // matrix non-zero values
vector<uint16_t> neighbor; // matrix non-zero values

// implement the graph dataflow on a single core
// |___|___|
// |___|___|
// |___|___|
// TODO: padding, fix done (maybe involve all 4 vals, use both edge dest and
// val)
// Another difference is that it is dense vector, decide dataset size
// Should we exploit the fact that there is no point of weights? TODO: we can
// just count that const here -- real data to be read is the earlier pr value,
// and dest vertex id
void mv(long tid) {
  
  // this should work on columns in range of tid, and a tile of vector
  int start_col = tid*(V/NUM_THREADS);
  int end_col = (tid+1)*(V/NUM_THREADS);

  // cout << "Start col: " << start_col << " and col: " << end_col << endl;
  /*if(tid==1) {
    SS_WAIT_ALL();
    return;
  }*/
  
  SS_VREPEAT_PORT(P_pr_row_size2);
  SS_DMA_READ(&prev_vertex_data[0], 2, 2, V+1, P_pr_prev_vert_pr);

  // TODO: add later
  // SS_VREPEAT_PORT(P_pr_row_size3);
  // SS_CONST(P_pr_R, 0, V);

  // last should be V+1-V, ... V-V-1,...1..0
  SS_CONST(P_pr_offset_list0,offset[0],1);

  SS_ADD_PORT(P_pr_offset_list0);
  SS_DMA_READ(&offset[1], 2, 2, V, P_pr_offset_list1);

  SS_CONST(P_pr_offset_list1,offset[V+1],1);

  // edge weight can be calculated inside dfg
  SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
  SS_INDIRECT_2D(P_pr_start_ind, &neighbor[0], V+1, 2, 2, P_pr_row_size1, P_pr_dest_id);

  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  // for multicore, only change is that it has to do remote update
  SS_ATOMIC_SCR_OP(P_pr_addr, P_pr_val, 0, neighbor.size(), 0);  
  // SS_ATOMIC_SCR_OP(P_pr_addr, P_pr_val, getRemoteBankedOffset(1,0,1), neighbor.size(), 0);  
 
  uint16_t x;
  SS_RECV(P_pr_done, x);
  SS_RESET();
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
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    uint16_t src, dst, wgt;
    // char ignore;
    // iss >> src >> dst; //  >> wgt;
// FOR INITIAL 1-based graph
    // iss >> dst >> src; //  >> wgt;
    // src = src-1; dst = dst-1;
// FOR CSR
    iss >> src >> dst; 
    // cout << src << " " << dst << endl;
    ++e;
    // cout << "Neighbor at e: " << e << " is: " << neighbor[e] << endl;
    if(src!=prev_v) {

      // Padding here
      if(prev_v==-1) {
        prev_col_size=0;
      } else {
        prev_col_size = e - offset[prev_v];
      }
      // cout << "prev_v: " << prev_v << " and col size: " << prev_col_size << endl;
      pad_size = (int)prev_col_size%4;
      if(pad_size!=0) {
        pad_size = 4 - pad_size;
        // cout << "PAD SIZE: " << pad_size << " when dest id: " << dst << endl;
        for(uint16_t k=0; k<pad_size; ++k) {
          neighbor.push_back(0);
        }
        e += pad_size;
      }

      offset[prev_v+1]=e;
      // cout << (prev_v+1) << " OFFSET: " << e << endl;
      int k=prev_v+1;
      while(offset[--k]==0 && k>0) {
        offset[k]=prev_offset;
      }
      prev_offset=e;
      prev_v=src;
    }
    neighbor.push_back(dst);
    // cout << _neighbor[e].wgt << " " << _neighbor[e].dst_id << " " << _offset[prev_v-1] << endl;
  }
  // offset[V] = E;
  
  // cout << "AFTER LAST EDGE E IS: " << e << endl;
  // Padding here
  // prev_col_size = e - offset[prev_v];
  // FIXME: confirm this
  pad_size = (int)neighbor.size()%4;
  if(pad_size!=0) {
    pad_size = 4 - pad_size;
    for(uint16_t k=0; k<pad_size; ++k) {
      neighbor.push_back(0);
    }
    e += pad_size;
  }
  // cout << "FINAL NEIGHBOR SIZE: " << neighbor.size() << endl; 
  // offset[V] = e;
  offset[V] = neighbor.size();
  int k=V;
  while(offset[--k]==0 && k>0) { // offset[0] should be 0
    offset[k]=prev_offset;
  }
  fclose(graph_file);
  cout << "Done reading graph file!\n";
}

void *entry_point(void *threadid) {

  long tid;
  tid = (long)threadid;
  // cout << "Before synch came here for tid: " << tid << endl;
  
  // Synchronization point
  int rc = pthread_barrier_wait(&barr);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
  {
    printf("Could not wait on barrier\n");
    // exit(-1);
  }

  // cout << "After synch came here for tid: " << tid << endl;

  begin_roi();
  SS_CONFIG(pr_config, pr_size);
  mv(tid);
  end_roi();
  sb_stats();
  pthread_barrier_wait(&barr2);
  return NULL;
}
 
void init_prev_pr() {
  for(int i=0; i<V; ++i) {
    prev_vertex_data[i] = 1;
  }
}

void pad_both_at_end() {
  // prev_vertex_data[V] = 0;
  prev_vertex_data[V] = V;
  offset[V+1] = offset[V]+4;
  for(int i=0; i<4; ++i) {
    // neighbor.push_back(0);
    neighbor.push_back(1);
  }
}

void print_neighbor() {
  for(int i=0; i<V+2; ++i) {
    cout << "Index pointer at column i: " << i << " is: " << offset[i] << endl;
  }
  /*for(unsigned i=0; i<neighbor.size(); ++i) {
    cout << "Neighbor at i: " << neighbor[i] << endl;
  }*/
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
 
  pthread_t threads[NUM_THREADS];
  // pthread_t threads[NUM_THREADS+1]; // last one is the dummy thread for memory broadcast
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
      return 0;
    }
  }
  return 0;
}
