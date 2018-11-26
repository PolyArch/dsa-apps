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

#define NUM_THREADS 2

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
  vector<uint64_t> inst_id;
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

  vector<uint64_t> inst_id;
  struct SplitInfo init_info = {0, 0, 0.0};

  TNode temp = {inst_id, nullptr, nullptr, hists, init_info};
  return temp;
}

// TODO: check this!
void mapping(struct TNode* node) {
  SB_WAIT_DF(0, 0);
  unsigned n = node->inst_id.size();
  // map dfg
  
  // calculated from reduction
  int feat_id=0;
  int split_thres=k/2;
  
  // extra read of id's: do something
  // replicated from histogram building (keep it in linear scratchpads?)
  SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
  SB_CONFIG_INDIRECT(T32, T32, M+2);
  SB_INDIRECT(P_IND_1, &fixed_inst_feat[0][feat_id], n, P_map_feat_val);

  SB_CONST(P_map_split_thres, split_thres, n);

  // What is this?
  SB_CONST(P_map_child1_offset, 0, n);
  SB_CONST(P_map_child2_offset, 64, n);
  SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_map_node_id);

  SB_CONFIG_ATOMIC_SCR_OP(T32, T32, T32);
  // SB_ATOMIC_SCR_OP(P_map_offset, P_IND_DOUB1, 0, n, 3); // I want just update
  SB_ATOMIC_SCR_OP(P_map_offset, P_map_id, 0, n, 3); // I want just update
 
  // indirect wr to only scr possible: not allowed to use output port as
  // indirect port? (weird): should i remove that condition?
  // SB_INDIRECT_WR(P_map_offset, 0, n, P_map_id); // it is not mapping correct value as the output...need to see
  SB_WAIT_ALL();


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
  SB_CONST(P_local_redn_prev_err, 100000, 1);
  SB_STRIDE(0,0);
  SB_RECURRENCE(P_local_redn_reduced_error, P_local_redn_prev_err, NUM_THREADS-1);
  // SB_SCR_WRITE(P_local_redn_reduced_error, 4, k*4*12 + 4*4);
  SB_REM_SCRATCH(0, 0, 4, 1, P_local_redn_reduced_error, 0);
  SB_WAIT_ALL();
}

// find min of 4 (get on control core only)
void local_reduction2() {

  uint64_t mask=0;
  addDest(mask, 0); // 0 is the central core

  // TODO: get the optimal local <feat, thres> pair
  SB_SCRATCH_READ(k*4*12, 4*4, P_local_redn_cur_err);
  SB_CONST(P_local_redn_prev_err, 100000, 1);
  SB_STRIDE(0,0);
  SB_RECURRENCE(P_local_redn_reduced_error, P_local_redn_prev_err, 3);
  // TODO: this should be remote scr_wr for global reduction
  SB_REM_PORT(P_local_redn_reduced_error, 1, mask, P_local_redn_cur_err);
  // SB_SCR_WRITE(P_local_redn_reduced_error, 4, k*4*12 + 4*4);
  SB_WAIT_ALL();
}

void local_reduction1() {
  // calculate the error from the three histograms for all 4 features
  // all 16-bit values (16*12=192 bits)

  // TODO: get the optimal thres for each feature here
  for(int i=0; i<4; ++i) {
    // scratch read 3 histograms
    SB_SCR_PORT_STREAM(0+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_C);
    SB_SCR_PORT_STREAM(k*4+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_G);
    SB_SCR_PORT_STREAM(k*4*2+i*k*4*3, sizeof(uint32_t), sizeof(uint32_t), k, P_local_redn_H);

    // send constant
    SB_DMA_READ(&fm[0], 4, 4, k, P_local_redn_F); // that f values

    // deal with outputs
    SB_CONST(P_local_redn_const,2,k-1);
    SB_CONST(P_local_redn_const,1,1);
    SB_SCR_WRITE(P_local_redn_final_error, 4, k*4*12 + i*4);
  }
  SB_WAIT_ALL();
}

void build_histogram(long j, struct TNode* node) {
 
  uint64_t feature_offset = merge_bits(0,k*4*3,k*4*3*2,k*4*3*3);

  uint64_t local_offset1 = merge_bits(k*4, k*4, k*4, k*4);
  uint64_t local_offset2 = merge_bits(k*4*2, k*4*2, k*4*2, k*4*2);
  
  uint64_t local_offset3 = merge_bits(k*4+k*4*3*4, k*4+k*4*3*4, k*4+k*4*3*4, k*4+k*4*3*4);
  uint64_t local_offset4 = merge_bits(k*4*2+k*4*3*4, k*4*2+k*4*3*4, k*4*2+k*4*3*4, k*4*2+k*4*3*4);

  unsigned n = node->inst_id.size();

  // inst_id could be just 16-bit (this could be reduced, also it would need
  // padding then for n)

  SB_ADD_PORT(P_IND_2);
  SB_ADD_PORT(P_IND_3);
  SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);

  // _index_addr + index * _ind_mult + _offsets[_index_in_offsets]*_data_bytes;
  SB_CONFIG_INDIRECT(T64, T64, M*sizeof(uint16_t)); // FIXME: this multiplier is good enough?
  SB_INDIRECT(P_IND_1, &fixed_inst_feat[0][j], n, P_gbdt_A);

  SB_CONFIG_INDIRECT(T64, T32, sizeof(uint32_t));
  SB_INDIRECT(P_IND_2, &hess[0], n, P_gbdt_hess);

  SB_CONFIG_INDIRECT(T64, T32, sizeof(uint32_t));
  SB_INDIRECT(P_IND_3, &grad[0], n, P_gbdt_grad);

  // SB_CONST(P_gbdt_const, 1, n);
  SB_CONST(P_gbdt_const, 1, n);

  SB_CONST(P_gbdt_local_offset1, local_offset1, n);
  SB_CONST(P_gbdt_local_offset2, local_offset2, n);
  SB_CONST(P_gbdt_local_offset3, local_offset3, n);
  SB_CONST(P_gbdt_local_offset4, local_offset4, n);

  SB_CONFIG_ATOMIC_SCR_OP(T16, T32, T32);
  // iters is num of ops
  // FIXME: CHECKME: I want it to be 16-bits -- should follow address datatype
  SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, feature_offset, n*8*3, 0);
  SB_WAIT_SCR_WR();
  SB_WAIT_ALL();
}
/*
// Let's do 8 features at a time
void build_histogram(long j, struct TNode* node) {
  part1 = k*4*4;
  part1 = k*4*2*4;
  uint64_t local_offset1 = merge_bits(part1, part1, part1, part1);
  uint64_t local_offset2 = merge_bits(part2, part2, part2, part2);
  uint64_t fused_const = merge_bits(1, 1, 1, 1);
  uint64_t feature_offset = merge_bits(0,k*4*3,k*4*3*2,k*4*3*3);
  local_offset1 = merge_bits(k*4, k*4, k*4, k*4);
  local_offset2 = merge_bits(k*4*2, k*4*2, k*4*2, k*4*2);
  unsigned n = node->inst_id.size();

  // inst_id could be just 16-bit (this could be reduced, also it would need
  // padding then for n)

  SB_ADD_PORT(P_IND_2);
  SB_ADD_PORT(P_IND_3);
  SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);

  // _index_addr + index * _ind_mult + _offsets[_index_in_offsets]*_data_bytes;
  SB_CONFIG_INDIRECT(T64, T64, M*sizeof(uint16_t)); // FIXME: this multiplier is good enough?
  SB_INDIRECT(P_IND_1, &fixed_inst_feat[0][j], n, P_gbdt_A);

  SB_CONFIG_INDIRECT(T64, T32, sizeof(uint32_t));
  SB_INDIRECT(P_IND_2, &hess[0], n, P_gbdt_hess);

  SB_CONFIG_INDIRECT(T64, T32, sizeof(uint32_t));
  SB_INDIRECT(P_IND_3, &grad[0], n, P_gbdt_grad);

  // SB_CONST(P_gbdt_const, 1, n);
  SB_CONST(P_gbdt_const, fused_const, n);

  SB_CONST(P_gbdt_local_offset1, local_offset1, n);
  SB_CONST(P_gbdt_local_offset2, local_offset2, n);

  // SB_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
  // addr, val, out
  SB_CONFIG_ATOMIC_SCR_OP(T16, T32, T32);
  // iters is num of ops
  // SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, offset, n*4*3, 0);
  // FIXME: CHECKME: I want it to be 16-bits -- should follow address datatype
  SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, feature_offset, n*4*3, 0);
  SB_WAIT_SCR_WR();
  SB_WAIT_ALL();
}
*/
void hierarchical_reduction(long tid) {
  uint64_t mask=0;
  for(int stride=1; stride<4; ++stride) {
    for(int i=0; i<8; i+=stride) {
      if(i%2 && tid>0) {
        addDest(mask, tid-1);
        SB_SCR_REM_PORT(k*4*3, 3, mask, P_local_redn_prev_err);
      } else {
        SB_WAIT_DF(3*4, 0);
        SB_SCRATCH_READ(k*4*3, 4*3, P_local_redn_cur_err);
        SB_SCR_WRITE(P_local_redn_reduced_error, 4*3, k*4*3);
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

  begin_roi();
  SB_CONFIG(gbdt_config, gbdt_size);
  build_histogram(tid, cur_node);
  SB_CONFIG(local_redn_config, local_redn_size);
  local_reduction1();
  local_reduction2();
  global_reduction(tid);
  // hierarchical_reduction(tid);
  // SB_CONFIG(map_config, map_size);
  // mapping(cur_node);
  end_roi();
  sb_stats();
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

    // begin_roi();
    // build_histogram(0, node);

    // cout << "Done with histogram building\n";

    // // local_reduction1();
    // // local_reduction2();
    // // cout << "Done with local reduction\n";
    // // hierarchical reduction
    // // broadcast of the value
    // // mapping

    // end_roi();
    // sb_stats();

    cur_node = node;

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
  int n=0;
  while(fgets(lineToRead, 5000, train_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    char ignore;
	int index;

	iss >> inst_label[n];

    for(int i=0; i<M; i++){
	  // iss >> index >> ignore >> tree.inst_feat[n][i];
      // iss >> index >> ignore >> tree.fixed_inst_feat[n][i];
      iss >> fixed_inst_feat[n][i];
	  // tree.fixed_inst_feat[n][i] = DOUBLE_TO_FIX(tree.inst_feat[n][i]);
	}
	n++;
  }

  fclose(train_file);
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

  vector<uint64_t> inst_id;
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
