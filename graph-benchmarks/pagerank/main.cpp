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
// FIXME: now difference doesn't seem correct
void mv(long tid) {
  
  // 0..1676 (both included), 1677..3353
  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
  // end_col-start_col is V here
  // start_col = tid*(EFF_VERT_PER_THREAD+1);
  // end_col = (tid+1)*(EFF_VERT_PER_THREAD+1); // not sure if correct

  // cout << "Start col: " << start_col << " and col: " << end_col << endl;
  // cout << "offset1: " << offset[start_col] << " offset2: " << offset[end_col] << " size of edges: " << offset[end_col]-offset[start_col] << endl;
  
  SS_VREPEAT_PORT(P_pr_row_size2);
  SS_DMA_READ(&prev_vertex_data[start_col], 2, 2, end_col-start_col, P_pr_prev_vert_pr);

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
  // SS_INDIRECT_2D(P_pr_start_ind, &neighbor[offset[start_col]], end_col-start_col+1, 2, 2, P_pr_row_size1, P_pr_dest_id);
  // index should be taken care of by indices in the offset...
  SS_INDIRECT_2D(P_pr_start_ind, &neighbor[0], end_col-start_col, 2, 2, P_pr_row_size1, P_pr_dest_id);

  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  // for multicore, only change is that it has to do remote update
  SS_ATOMIC_SCR_OP(P_pr_addr, P_pr_val, 0, offset[end_col]-offset[start_col], 0);  
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
  int part=0;
  bool pad_phase=false;
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
    // cout << "Read tuple: " << src << " " << dst << endl;
    ++e;
    // cout << "Neighbor at e: " << e << " is: " << neighbor[e] << endl;
    
    // FIXME: add for padding in between them -- can bother for later
    if(src!=0 && pad_phase && src > 10) { // padded value: after a partition
      // cout << "INSERTING OFFSET AT VERTEX: " << (prev_v+1+part) << " and the value of number of edges till now: " << e << endl;
      ++part;
      offset[prev_v+1+part] = e;
      // neighbor.push_back(dst);
      prev_v = src;
      pad_phase = false;
    } else if(src!=prev_v && !pad_phase) {

      // Padding here
      if(prev_v==-1) {
        prev_col_size=0;
      } else {
        prev_col_size = e - offset[prev_v+part];
      }

      // cout << "prev_v: " << prev_v << " and col size: " << prev_col_size << endl;
      pad_size = (int)prev_col_size%4;
      if(pad_size!=0) {
        pad_size = 4 - pad_size;
        // cout << "PAD SIZE: " << pad_size << " when dest id: " << dst << endl;
        for(int k=0; k<pad_size; ++k) {
          neighbor.push_back(0);
        }
        e += pad_size;
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
  offset[V+2*(NUM_THREADS-1)] = neighbor.size();
  int k=V+2*(NUM_THREADS-1);
  while(offset[--k]==0 && k>0) { // offset[0] should be 0
    offset[k]=prev_offset;
  }
  // cout << "Offset at 1: " << offset[1] << endl;
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
