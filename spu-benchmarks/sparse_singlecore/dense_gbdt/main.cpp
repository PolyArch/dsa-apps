#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "gbdt.dfg.h"
#include "local_redn.dfg.h"
// #include "test.dfg.h"
// #include "final_redn.dfg.h"
#include "map.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <sstream>
#include <string>
#define iters 63
#define k 64

#define NUM_THREADS 8

using namespace std;

// Barrier variable
pthread_barrier_t barr;

struct SplitInfo {
  uint64_t split_feat_id;
  uint64_t thres;
  double entropy;
};

struct featInfo {
  double label_hist[64];
  double count_hist[64];
};

struct TNode {
  vector<uint16_t> inst_id;
  struct TNode* child1;
  struct TNode* child2;
  vector<featInfo> feat_hists;
  SplitInfo info;
};

uint64_t min_children;
uint64_t max_depth;
uint64_t depth = 0; // temp variable

// Data structures to read from file
float inst_feat[N][M];
uint32_t inst_label[N];

// Dynamic data structures
uint16_t fixed_inst_feat[N][M];
uint32_t hess[N];
uint32_t grad[N];
uint32_t fm[k];

uint16_t part1 = 16384/3;
uint16_t part2 = 16384*2/3;

void init_data() {
  min_children = 2;
  max_depth = 10;
}

TNode init_node(){
  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
    init_hists.label_hist[i] = 0.0;
    init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(uint64_t i=0; i<M; ++i) {
    hists.push_back(init_hists);
  }

  vector<uint16_t> inst_id;
  struct SplitInfo init_info = {0, 0, 0.0};

  TNode temp = {inst_id, nullptr, nullptr, hists, init_info};
  return temp;
}

// TODO: check this!
void mapping(struct TNode* node) {
  SS_WAIT_DF(0, 0);
  unsigned n = node->inst_id.size();
  // map dfg
  
  // calculated from reduction
  int feat_id=0;
  int split_thres=k/2;
  
  // extra read of id's: do something
  // replicated from histogram building (keep it in linear scratchpads?)
  SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
  SS_CONFIG_INDIRECT(T32, T32, M+2);
  SS_INDIRECT(P_IND_1, &fixed_inst_feat[0][feat_id], n, P_map_feat_val);

  SS_CONST(P_map_split_thres, split_thres, n);

  // What is this?
  SS_CONST(P_map_child1_offset, 0, n);
  SS_CONST(P_map_child2_offset, 64, n);
  SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_map_node_id);

  SS_CONFIG_ATOMIC_SCR_OP(T32, T32, T32);
  // SS_ATOMIC_SCR_OP(P_map_offset, P_IND_DOUB1, 0, n, 3); // I want just update
  SS_ATOMIC_SCR_OP(P_map_offset, P_map_id, 0, n, 3); // I want just update
 
  // indirect wr to only scr possible: not allowed to use output port as
  // indirect port? (weird): should i remove that condition?
  // SS_INDIRECT_WR(P_map_offset, 0, n, P_map_id); // it is not mapping correct value as the output...need to see
  SS_WAIT_ALL();


}

// everybody should send to core id 0 (sequence doesn't matter)
// portA: thres
// portB: feat_id
// portC: min_err
// reduction tree for the horizontal thing -- actually everyone sends to the
// central node seems better
void global_reduction(long tid) {
  if(tid!=0) return;
  
  // TODO: get the optimal local <feat, thres> pair
  SS_CONST(P_local_redn_prev_err, 100000, 1);
  SS_STRIDE(0,0);
  SS_RECURRENCE(P_local_redn_reduced_error, P_local_redn_prev_err, NUM_THREADS-1);
  // SS_SCR_WRITE(P_local_redn_reduced_error, 4, k*4*12 + 4*4);
  SS_REM_SCRATCH(0, 0, 4, 1, P_local_redn_reduced_error, 0);
  SS_WAIT_ALL();
}

// find min of 4 (get on control core only)
void local_reduction2() {

  uint64_t mask=0;
  addDest(mask, 0); // 0 is the central core

  // TODO: get the optimal local <feat, thres> pair
  SS_SCRATCH_READ(k*4*12, 4*4, P_local_redn_cur_err);
  SS_CONST(P_local_redn_prev_err, 100000, 1);
  SS_STRIDE(0,0);
  SS_RECURRENCE(P_local_redn_reduced_error, P_local_redn_prev_err, 3);
  // TODO: this should be remote scr_wr for global reduction
  SS_REM_PORT(P_local_redn_reduced_error, 1, mask, P_local_redn_cur_err);
  // SS_SCR_WRITE(P_local_redn_reduced_error, 4, k*4*12 + 4*4);
  SS_WAIT_ALL();
}

void local_reduction1() {
  // calculate the error from the three histograms for all 4 features
  // all 16-bit values (16*12=192 bits)

  // TODO: get the optimal thres for each feature here
  for(int i=0; i<4; ++i) {
    // scratch read 3 histograms
    SS_SCR_PORT_STREAM(0+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_C);
    SS_SCR_PORT_STREAM(k*4+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_G);
    SS_SCR_PORT_STREAM(k*4*2+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_H);

    // send constant
    SS_DMA_READ(&fm[0], 4, 4, k, P_local_redn_F); // that f values

    // deal with outputs
    SS_CONST(P_local_redn_const,2,k-1);
    SS_CONST(P_local_redn_const,1,1);
    SS_SCR_WRITE(P_local_redn_final_error, 4, k*4*12 + i*4);
  }
  SS_WAIT_ALL();
}

void build_histogram(long tid, struct TNode* node) {
 
  // just take into consideration the elements here
  uint64_t feature_offset = merge_bits(0,k*3,k*3*2,k*3*3);

  uint64_t local_offset1 = merge_bits(k, k, k, k);
  uint64_t local_offset2 = merge_bits(k*2, k*2, k*2, k*2);

  unsigned n = node->inst_id.size();
  n = (n/4)*4; // padding required because of indirect ports

  // TODO: broadcast this!
  SS_ADD_PORT(P_IND_2);
  SS_ADD_PORT(P_IND_3);
  // SS_RECURRENCE(P_IND_4, P_IND_1, n/4);
  SS_DMA_READ(&node->inst_id[0], 8, 8, n/4, P_IND_1);

  // _index_addr + index * _ind_mult + _offsets[_index_in_offsets]*_data_bytes;
  // itype, dtype, mult
  // number of features per core...
  SS_CONFIG_INDIRECT1(T16, T64, 8*sizeof(uint16_t), 1);
  SS_INDIRECT(P_IND_1, &fixed_inst_feat[0][0], n, P_gbdt_A);
  // SS_INDIRECT_SCR(P_IND_1, getLinearAddr(0), n, P_gbdt_A);

  SS_CONFIG_INDIRECT(T16, T32, sizeof(uint32_t));
  SS_INDIRECT(P_IND_2, &hess[tid*NUM_THREADS], n, P_gbdt_hess);

  SS_CONFIG_INDIRECT(T16, T32, sizeof(uint32_t));
  SS_INDIRECT(P_IND_3, &grad[tid*NUM_THREADS], n, P_gbdt_grad);

  SS_CONST(P_gbdt_const, 1, n);
  SS_CONST(P_gbdt_local_offset1, local_offset1, n);
  SS_CONST(P_gbdt_local_offset2, local_offset2, n);
  // SS_CONST(P_gbdt_local_offset3, local_offset3, n);
  // SS_CONST(P_gbdt_local_offset4, local_offset4, n);

  // FIXME: it repeats at the granularity of whole vec port I guess -- might
  // need to change offsets for correct result
  SS_REPEAT_PORT(4);
  SS_RECURRENCE(P_gbdt_D, P_gbdt_dummy_in, n*3);

  // n*8*3
  SS_CONFIG_ATOMIC_SCR_OP(T16, T32, T32);
  SS_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_dummy_out, feature_offset, n*4*3*2, 0);
  SS_WAIT_ALL();
}

void broadcast_from_cc(struct TNode* node) {

  uint64_t mask=0;
  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(mask, i);
  }
  unsigned n = node->inst_id.size();
  n = (n/4)*4; // padding required because of indirect ports
  SS_DMA_READ(&node->inst_id[0], 8, 8, n/4, P_IND_1);
  SS_REM_PORT(P_IND_1, n/4, mask, P_IND_4);
  SS_WAIT_ALL();
}


void hierarchical_reduction(long tid) {
  uint64_t mask=0;
  for(int stride=1; stride<4; ++stride) {
    for(int i=0; i<8; i+=stride) {
      if(i%2 && tid>0) {
        addDest(mask, tid-1);
        SS_SCR_REM_PORT(k*4*3, 3, mask, P_local_redn_prev_err);
      } else {
        SS_WAIT_DF(3*4, 0);
        SS_SCRATCH_READ(k*4*3, 4*3, P_local_redn_cur_err);
        SS_SCR_WRITE(P_local_redn_reduced_error, 4*3, k*4*3);
      }
    }
  }
}

struct TNode* cur_node;
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


  // load all instances of the features (8-features per core)
  SS_CONFIG(gbdt_config, gbdt_size);
  // FIXME: check here!
  // SS_DMA_SCRATCH_LOAD(&fixed_inst_feat[0][tid], 2, 2, N*8, getLinearAddr(0));
  // SS_WAIT_ALL();
  build_histogram(tid, cur_node);
  begin_roi();
  build_histogram(tid, cur_node);
  // SS_CONFIG(local_redn_config, local_redn_size);
  // local_reduction1();
  // local_reduction2();
  // global_reduction(tid);
  // hierarchical_reduction(tid);
  // SS_CONFIG(map_config, map_size);
  // mapping(cur_node);
  end_roi();
  sb_stats();
  // pthread_barrier_wait(&barr);
  return NULL;
}

void build_tree(struct TNode* node){
  // cout << "Came here to build tree" << "\n";
  double max_entr;
  uint64_t n;
  TNode child1[100];
  TNode child2[100];

  vector<TNode*> cur_nodes;
  cur_nodes.push_back(node);
  // cout << "Current nodes in queue are: " << cur_nodes.size() << " depth is: " << depth << "\n";
  for(unsigned node_id=0; node_id<cur_nodes.size() && depth < max_depth; ++node_id){
    // cout << "Node_id we are working on: " << node_id << "\n";
    max_entr = 0.0;
    // malloc--
    node = cur_nodes[node_id];
    child1[node_id] = init_node();
    child2[node_id] = init_node();
    node->child1 = &child1[node_id];
    node->child2 = &child2[node_id];

    n = node->inst_id.size();
    node->child1->inst_id.resize(n);
    node->child2->inst_id.resize(n);
    cout << "Starting process for a node\n";

    SS_CONFIG(gbdt_config, gbdt_size);
    // FIXME: check here!
    // SS_DMA_SCRATCH_LOAD(&fixed_inst_feat[0][0], 2, 2, N*8, getLinearAddr(0));
    // SS_WAIT_ALL();
 
    begin_roi();
    build_histogram(0, node);

    // cout << "Done with histogram building\n";

    // // local_reduction1();
    // // local_reduction2();
    // // cout << "Done with local reduction\n";
    // // hierarchical reduction
    // // broadcast of the value
    // // mapping

    end_roi();
    sb_stats();

    cur_node = node;

    /*
    broadcast_from_cc(node);
    // FIXME: ideally should create outside and put a global barrier here
    assert(NUM_THREADS<C);
    
    // Barrier initialization
    if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
    {
      printf("Could not create a barrier\n");
      return;
    }

    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    for(t=0;t<NUM_THREADS;t++){
      printf("In main: creating thread %ld\n", t);
      rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);     
      // rc = pthread_create(&threads[t], NULL, tree.build_tree, (void *)t);     
      if (rc){
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        return;
      }
    }
    
    for(int i = 0; i < NUM_THREADS; ++i) {
      if(pthread_join(threads[i], NULL)) {
    	printf("Could not join thread %d\n", i);
        return;
      }
    }
    */
  }
}

struct TNode* root;

int main() {

  init_data();

  string str(file);

  // FILE* train_file = fopen("datasets/binned_small_mslr.train", "r");
  // FILE* train_file = fopen("datasets/very_small.data", "r");
  FILE* train_file = fopen(str.c_str(), "r");

  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  cout << "Start reading train file!\n";
  /*
  int n=0;
  while(fgets(lineToRead, 5000, train_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    char ignore;
	int index;

	iss >> inst_label[n];

    while(iss >> index) {
      iss >> ignore >> fixed_inst_feat[n][index];
    }
  n++;
  }

  fclose(train_file);
  */
  
  for(int i=0; i<N; ++i){
    for(int j=0; j<M; ++j) {
      fixed_inst_feat[i][j]=rand()%64;
    }
  }
  
  cout << "Done reading file!\n";

  for(int i=0; i<k; ++i){
	fm[i]=i;
  }
  // tree.feature_binning();

  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
    init_hists.label_hist[i] = 0.0;
    init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(uint64_t i=0; i<M; ++i) {
    hists.push_back(init_hists);
  }

  vector<uint16_t> inst_id;
  for(unsigned i=0; i<N; i++) {
	inst_id.push_back(i);
  }

  // initialize hess,grad variables here
  for(int i=0; i<N; i++) {
	hess[i] = DOUBLE_TO_FIX(float(i*0.2));
	grad[i] = DOUBLE_TO_FIX(float(i*0.3));
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  root = &node;

  build_tree(root);
  return 0;
}
