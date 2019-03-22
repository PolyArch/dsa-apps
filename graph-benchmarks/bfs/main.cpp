#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "bfs.dfg.h"
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

// resparsification: dynamic? (let's assume phase 2 for now and do static
// thing)
vector<uint16_t> cur_dist_val; // output vector
vector<uint16_t> cur_dist_ind; // output vector

uint16_t cur_vertex_data[V+NUM_THREADS]; // output vector

// TODO: change this to sparse: ind would be active bitvector
uint16_t prev_vertex_data[V+NUM_THREADS]; // input vector
// TODO: this even needs to be tiled for multi-core (same as the tile of dense
// output vector stored in each CGRA)
vector<uint16_t> prev_dist_val[NUM_THREADS]; // input vector
vector<uint16_t> prev_dist_ind[NUM_THREADS]; // input vector

uint16_t offset[V+2*(NUM_THREADS-1)+1]; // matrix ptr
vector<uint16_t> neighbor; // matrix non-zero values

bool spu_resparsify() {
  // read from banked scratchpad and send it back -- this can be parallelized
  // I guess -- but we will first implement separate
  return false;
}

void mv(long tid) {

  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
  bool should_iter=false;

  do {

    unsigned num_active_vert_per_core = prev_dist_ind[tid].size();
    // cout << "Active vertex for tid: " << tid << " and number of active vertex: " << num_active_vert_per_core << endl;
    
    SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
    SS_ATOMIC_SCR_OP(P_bfs_addr, P_bfs_val, 0, offset[end_col]-offset[start_col], 2);  
 
    // reading column/prev_vert_dist of active vertices
    SS_DMA_READ(&prev_dist_val[tid][0], 2, 2, num_active_vert_per_core, P_bfs_pass1);
    SS_VREPEAT_PORT(P_bfs_row_size2);
    SS_RECURRENCE(P_bfs_pass2, P_bfs_prev_vert_dist, num_active_vert_per_core);

    // reading column/prev_vert_ind of active vertices
    SS_DMA_READ(&prev_dist_ind[tid][0], 2, 2, num_active_vert_per_core, P_IND_1);
    SS_CONFIG_INDIRECT1(T16,T16,2,1);
    SS_INDIRECT(P_IND_1, &offset[0], num_active_vert_per_core, P_bfs_offset_list);

    SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
    SS_INDIRECT_2D(P_bfs_start_ind, &neighbor[0], num_active_vert_per_core, 2, 2, P_bfs_row_size1, P_bfs_dest_id);

    uint16_t x;
    SS_RECV(P_bfs_done, x);
    SS_RESET();

    SS_GLOBAL_WAIT();
    SS_WAIT_ALL();

    should_iter = spu_resparsify();

  } while(should_iter);
}

void mv_complete(long tid) {

  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
  bool should_iter=false;

  unsigned num_active_vert_per_core = prev_dist_ind[tid].size();
  // first send the pre-initialized stuff
  SS_DMA_READ(&prev_dist_val[tid][0], 2, 2, num_active_vert_per_core, P_bfs_pass1);
  SS_DMA_READ(&prev_dist_ind[tid][0], 2, 2, num_active_vert_per_core, P_IND_1);

  do {

    num_active_vert_per_core = prev_dist_ind[tid].size();
    // cout << "Active vertex for tid: " << tid << " and number of active vertex: " << num_active_vert_per_core << endl;
    
    SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
    SS_ATOMIC_SCR_OP(P_bfs_addr, P_bfs_val, 0, offset[end_col]-offset[start_col], 2);  
 
    // reading column/prev_vert_dist of active vertices
    SS_VREPEAT_PORT(P_bfs_row_size2);
    SS_RECURRENCE(P_bfs_pass2, P_bfs_prev_vert_dist, num_active_vert_per_core);

    // reading column/prev_vert_ind of active vertices
    SS_CONFIG_INDIRECT1(T16,T16,2,1);
    SS_INDIRECT(P_IND_1, &offset[0], num_active_vert_per_core, P_bfs_offset_list);

    SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
    SS_INDIRECT_2D(P_bfs_start_ind, &neighbor[0], num_active_vert_per_core, 2, 2, P_bfs_row_size1, P_bfs_dest_id);

    uint16_t x;
    SS_RECV(P_bfs_done, x);
    SS_RESET();

    SS_GLOBAL_WAIT();
    SS_WAIT_ALL();

    // read all vertices -- let's use linear scratchpad for old_dist
    // but we will need to move data from linear to banked scratchpad
    // SS_SCRATCH_READ(0, 2*V/NUM_THREADS, P_bfs_old_dist);
    // SS_SCRATCH_READ(0, 2*V/NUM_THREADS, P_bfs_old_dist);
    //
    // SS_RECURRENCE(P_bfs_prev_dist, P_bfs_prev_dist_val, 1000); // might have
    // to reset
    // SS_RECURRENCE(P_bfs_prev_ind, P_IND_1, 1000); // need to fix variable
    // bit widths

    should_iter = spu_resparsify();

  } while(should_iter);
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
  SS_CONFIG(bfs_config, bfs_size);
  mv(tid);
  end_roi();
  sb_stats();

  cout << "Returned back with tid: " << tid << endl;

  pthread_barrier_wait(&barr2);

  return NULL;
}
 
// only set active vertices (some of them to 1 here)
void init_prev_dist() {
  for(int i=0; i<V+NUM_THREADS; ++i) {
    prev_vertex_data[i] = 1;
  }
}

// FIXME: these numbers have to made scalable
// this should be done after every iteration
// sparsify and tile among all threads
void preprocess_prev_dist() {
  int offset=0;
  for(int i=0; i<NUM_THREADS; ++i) {
    // for(int j=0; j<V/NUM_THREADS; ++j) {
    for(int j=0; j<10; ++j) {
      prev_dist_val[i].push_back(1);
      prev_dist_ind[i].push_back(offset+j);
      /*if(prev_vertex_data[i]!=0) {
        prev_dist_val[i].push_back(prev_vertex_data[i]);
        prev_dist_ind[i].push_back(i*NUM_THREADS+j);
      }*/
    }
    // prev_dist_ind[i].push_back((i+1)*V/NUM_THREADS+NUM_THREADS-1);
    prev_dist_val[i].push_back(V);
    // offset=0, offset=1677
    offset = (i+1)*V/NUM_THREADS+NUM_THREADS-1;
    prev_dist_ind[i].push_back(offset-1+i); // not sure about end factor
    offset += 1;
  }
}

/*
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
*/

void print_neighbor() {
  for(int i=0; i<=V+2; ++i) {
    cout << "Index pointer at column i: " << i << " is: " << offset[i] << endl;
  }
  /*cout << "Address at neighbor 0: " << &neighbor[0] << endl;
  for(unsigned i=0; i<neighbor.size(); ++i) {
    cout << "Neighbor at i: " << neighbor[i] << endl;
  }*/
}

void print_cur_dist() {
  for(int i=0; i<NUM_THREADS; ++i) {
    for(unsigned j=0; j<prev_dist_val[i].size(); ++j) {
      cout << "Shortest distance of active vertex: " << prev_dist_ind[i][j] << " is: " << prev_dist_val[i][j] << endl;
    }
  }
}

int main() {
  read_input_file();
  // init_prev_dist(); // make sure this is not 0 (actually I should fix my sentinal problem)
  preprocess_prev_dist(); // this should not be needed ideally
  // pad_both_at_end();
  // print_neighbor();
  print_cur_dist();

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
