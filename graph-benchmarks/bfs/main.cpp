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

#define MAX_ITER 4

#define INF 65535

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

void mv(long tid) {

  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct

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

}

// initial old_dist is source with 0 dist and others with inf, put same thing
// in banked scratchpad
void preprocess_scratch_distance() {
  // SS_CONST_SCR(getLinearAddr(getLinearOffset(0,2)), 0, 1, 2);
  // SS_CONST_SCR(getLinearAddr(getLinearOffset(0,2))+2, INF, 4000, 2);
  SS_CONST_SCR(getBankedOffset(0,2), 0, 1, 2);
  SS_CONST_SCR(getBankedOffset(0,2)+2, INF, 4000, 2);
  SS_WAIT_ALL();
}

/*
 * Idea2: keep old_vert_dist dense version in both linear and banked scratchpad
 * For the apply phase, copy the sparse version in linear scratchpad (very slow
 * indirect writes -- possible?) and copy the data in the port
 * For the compute phase, it has sparse data in the port and dense already in
 * linear scratchpad
*/

void mv_complete(long tid) {

  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
  int iter=0;
  uint64_t active_list_size=0; // no active vertex at start
  
  begin_roi();

  if(tid==0) {
    active_list_size=1; // single source
  }
  // unsigned num_active_vert_per_core = prev_dist_ind[tid].size();
  
  // Just read const initial values to the first core
  if(tid==0) {
    SS_DCONST(P_bfs_pass1, 0, 1, T16);
    SS_DCONST(P_IND_1, 0, 1, T16);
  }

  do {

    SS_DCONST(P_bfs_pass1, V, 1, T16);
    SS_DCONST(P_IND_1, end_col+NUM_THREADS-3, 1, T16);
  
    // cout << "Active vertex for tid: " << tid << " and number of active vertex: " << num_active_vert_per_core << " and cur iter count: " << iter << endl;
    // cout << "Active vertex for tid: " << tid << " and cur iter count: " << iter << endl;
    
    SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
    SS_ATOMIC_SCR_OP(P_bfs_addr, P_bfs_val, 0, offset[end_col]-offset[start_col], 2);  
 
    // reading column/prev_vert_dist of active vertices
    SS_VREPEAT_PORT(P_bfs_row_size2);
    SS_RECURRENCE(P_bfs_pass2, P_bfs_prev_vert_dist, active_list_size+1);

    // reading column/prev_vert_ind of active vertices
    SS_CONFIG_INDIRECT1(T16,T16,2,1);
    SS_INDIRECT(P_IND_1, &offset[0], active_list_size+1, P_bfs_offset_list);

    SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
    SS_INDIRECT_2D(P_bfs_start_ind, &neighbor[0], active_list_size+1, 2, 2, P_bfs_row_size1, P_bfs_dest_id);

    uint16_t x;
    SS_RECV(P_bfs_done, x);
    // SS_DMA_WRITE(P_bfs_done, 2, 2, 1, &x); // had to garbage
    SS_RESET(); // should be that wait for outputs, clear all
    // SS_STREAM_RESET();

    // SS_GLOBAL_WAIT();
    SS_GLOBAL_WAIT(NUM_THREADS);
    // SS_WAIT_ALL();
    
    int num_dens_vert_per_core = V/NUM_THREADS;
    
    SS_SCRATCH_READ(0, 2*num_dens_vert_per_core, P_bfs_prev_dens_dist);
    
    // because 65535 is already infinity
    SS_DCONST(P_bfs_prev_dens_dist, 65534, 1, T16); // padding
    SS_DCONST(P_bfs_cur_iter, iter+1, num_dens_vert_per_core+1, T16); 
    
    SS_REPEAT_PORT(1);
    SS_RECURRENCE(P_bfs_prev_dist, P_bfs_pass1, num_dens_vert_per_core);
    SS_REPEAT_PORT(1);
    SS_RECURRENCE(P_bfs_prev_ind, P_IND_1, num_dens_vert_per_core);
    
    active_list_size=0;
    SS_RECV(P_bfs_active_list_size, active_list_size);

    SS_STREAM_RESET(); // this should be a barrier in itself -- but anyways
    SS_WAIT_STREAMS(); // wait until all streams are done -- not for CGRA (just wait for above one)
    // cout << "New active list size: " << active_list_size << endl;

    iter++;
  } while(iter!=MAX_ITER); // Ideally, complete active list should be empty
  // } while(active_list_size!=0);
  
  SS_RESET(); // to reset last sentinals
  SS_WAIT_ALL();

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
  int e=-1, prev_v=-1; // indices start from 0
  int prev_col_size=-1; int pad_size=-1;
  int part=0;
  bool pad_phase=false;
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    uint16_t src, dst;
    iss >> src >> dst; 
    // cout << src << " " << dst << endl;
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

  // begin_roi();
  SS_CONFIG(bfs_config, bfs_size);
  preprocess_scratch_distance();
  // mv(tid);
  mv_complete(tid);
  // end_roi();
  // sb_stats();

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
  // preprocess_prev_dist(); // this should not be needed ideally
  // pad_both_at_end();
  // print_neighbor();
  // print_cur_dist();
  // preprocess_scratch_distance();

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

// keep the sparse version in linear scratchpad and dense version in the
// banked scratchpad -- Is it good decision?
// For compute phase, we need old_vert_dist in sparse format, edge in dense
// format, new_vert_dist in dense (no need to rd)
// For apply phase, we need both old_vert_dist and new_vert_dist in dense
// format
/*
void mv_complete(long tid) {

  int start_col = tid*EFF_VERT_PER_THREAD;
  int end_col = (tid+1)*EFF_VERT_PER_THREAD; // not sure if correct
  bool should_iter=false;
  int iter=0;
  uint64_t active_list_size=1; // single source
  // unsigned num_active_vert_per_core = prev_dist_ind[tid].size();
  
  // first send the pre-initialized stuff -- read from linear scratchpad
  SS_SCRATCH_READ(getLinearAddr(getLinearOffset(0,2)), 2*active_list_size, P_bfs_prev_vert_dist);
  // FIXME: need to make sure that this datawidth works
  SS_SCRATCH_READ(getLinearAddr(getLinearOffset(1,2)), 2*active_list_size, P_IND_1);

  do {

    // num_active_vert_per_core = prev_dist_ind[tid].size();
    // cout << "Active vertex for tid: " << tid << " and number of active vertex: " << num_active_vert_per_core << " and cur iter count: " << iter << endl;
    cout << "Active vertex for tid: " << tid << " and cur iter count: " << iter << endl;
    
    SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
    SS_ATOMIC_SCR_OP(P_bfs_addr, P_bfs_val, 0, offset[end_col]-offset[start_col], 2);  
 
    // reading column/prev_vert_dist of active vertices
    SS_VREPEAT_PORT(P_bfs_row_size2);
    SS_RECURRENCE(P_bfs_pass2, P_bfs_prev_vert_dist, active_list_size);

    // reading column/prev_vert_ind of active vertices
    SS_CONFIG_INDIRECT1(T16,T16,2,1);
    SS_INDIRECT(P_IND_1, &offset[0], active_list_size, P_bfs_offset_list);

    SS_CONFIG_INDIRECT(T16,T16,2); // multiplier for offset
    SS_INDIRECT_2D(P_bfs_start_ind, &neighbor[0], active_list_size, 2, 2, P_bfs_row_size1, P_bfs_dest_id);

    uint16_t x;
    SS_RECV(P_bfs_done, x);
    SS_RESET();

    SS_GLOBAL_WAIT();
    SS_WAIT_ALL();

    int num_dens_vert_per_core = V/NUM_THREADS;
    SS_SCRATCH_READ(0, 2*num_dens_vert_per_core, P_bfs_prev_dens_dist);
    // SS_CONST(P_bfs_prev_dens_dist, 1, 1); // padding
    SS_CONST(P_bfs_prev_dens_dist, 0, 1); // padding
    SS_CONST(P_bfs_cur_iter, iter+1, num_dens_vert_per_core+1); 
    // Save it to linear scratchpad otherwise rd/wr problem?
    // SS_RECURRENCE(P_bfs_prev_dist, P_bfs_prev_vert_dist, num_dens_vert_per_core);
    // SS_RECURRENCE(P_bfs_prev_ind, P_IND_1, num_dens_vert_per_core);
    // TODO: write in linear scratchpad
    SS_SCR_WRITE(P_bfs_prev_dist, num_dens_vert_per_core*2, getLinearAddr(getLinearOffset(0,2)));
    SS_SCR_WRITE(P_bfs_prev_ind, num_dens_vert_per_core*2, getLinearAddr(getLinearOffset(1,2)));
    // can't i just wait on read streams to be done
    active_list_size=0;
    SS_RECV(P_bfs_active_list_size, active_list_size);
    // Oh this will clear the incoming ports as well -- so maybe let it work
    SS_RESET(); 
    // it got values 967, 556
    // cout << "New active list size: " << active_list_size << endl;

    SS_GLOBAL_WAIT(); // not sure, just not to create issues
    SS_WAIT_ALL();

    SS_CONST(P_bfs_prev_vert_dist, 1, 1);
    // TODO: may be wrong! because of endianness
    SS_2D_CONST(P_IND_1, 24, 1, 13, 1, 1);
    // SS_CONST(P_IND_1, 3352, 1);
    // SS_CONST(P_IND_1, 0, 1);
 

    // scratch read to the corresponding ports
    SS_SCRATCH_READ(getLinearAddr(getLinearOffset(0,2)), 2*active_list_size, P_bfs_prev_vert_dist);
    SS_SCRATCH_READ(getLinearAddr(getLinearOffset(1,2)), 2*active_list_size, P_IND_1);
    SS_GLOBAL_WAIT();
    SS_WAIT_ALL();

    iter++;
    should_iter = active_list_size!=0; // spu_resparsify();

  } while(should_iter);
}
*/
