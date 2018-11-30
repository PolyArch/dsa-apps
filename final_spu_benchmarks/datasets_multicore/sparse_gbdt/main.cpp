#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "sparse_none.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <sstream>
#include <string>
#define iters 63
#define k 64

#define NUM_THREADS 1

#define dummy_addr (640 | (640 << 16) | ((640 & 0xFFFFFFFFFFFFFFFF) << 32) | ((640 & 0xFFFFFFFFFFFFFFFF) << 48))

#define dummy_sentinal (SENTINAL16 | (SENTINAL16 << 16) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48))

#define node_sentinal (SENTINAL16 | (0 << 16) | ((0 & 0xFFFFFFFFFFFFFFFF) << 32) | ((0 & 0xFFFFFFFFFFFFFFFF) << 48))


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
// uint16_t fixed_inst_feat[N][M];
vector<uint16_t> feat_val[M];
vector<uint16_t> feat_ind[M];
uint32_t hess[N];
uint32_t grad[N];
uint32_t fm[k];
uint32_t y[N];

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

void build_histogram(long tid, struct TNode* node) {
 
  // just take into consideration the elements here
  uint64_t feature_offset = merge_bits(0,k*3,k*3*2,k*3*3);

  uint64_t local_offset1 = merge_bits(k, k, k, k);
  uint64_t local_offset2 = merge_bits(k*2, k*2, k*2, k*2);
  
  uint64_t local_offset3 = merge_bits(k+k*3*4, k+k*3*4, k+k*3*4, k+k*3*4);
  uint64_t local_offset4 = merge_bits(k*2+k*3*4, k*2+k*3*4, k*2+k*3*4, k*2+k*3*4);
    
  // FIXME
  uint64_t offset = merge_bits(0, k*2, k*4, k*6);
  uint64_t local_offset = merge_bits(k, k, k, k);

  unsigned n = node->inst_id.size();
  n = (n/4)*4; // padding required because of indirect ports

  cout << "Starting histogram building\n";



  // SB_CONFIG(sparse_none_config,sparse_none_size);
                
  SB_ADD_PORT(P_IND_1);
  SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_sparse_none_node_ind);

  SB_DMA_READ(&feat_ind[0][0], 2, 2, feat_ind[0].size(), P_sparse_none_feat_ind0);
  SB_DMA_READ(&feat_ind[1][0], 2, 2, feat_ind[1].size(), P_sparse_none_feat_ind1);
  SB_DMA_READ(&feat_ind[2][0], 2, 2, feat_ind[2].size(), P_sparse_none_feat_ind2);
  SB_DMA_READ(&feat_ind[3][0], 2, 2, feat_ind[3].size(), P_sparse_none_feat_ind3);

  SB_DMA_READ(&feat_val[0][0], 2, 2, feat_val[0].size(), P_sparse_none_feat_val0);
  SB_DMA_READ(&feat_val[1][0], 2, 2, feat_val[1].size(), P_sparse_none_feat_val1);
  SB_DMA_READ(&feat_val[2][0], 2, 2, feat_val[2].size(), P_sparse_none_feat_val2);
  SB_DMA_READ(&feat_val[3][0], 2, 2, feat_val[3].size(), P_sparse_none_feat_val3);

  // SB_CONST(P_sparse_none_feat_val, dummy_addr, 1);
  SB_CONST(P_sparse_none_feat_val0, 0, 1);
  SB_CONST(P_sparse_none_feat_val1, 0, 1);
  SB_CONST(P_sparse_none_feat_val2, 0, 1);
  SB_CONST(P_sparse_none_feat_val3, 0, 1);
  
  // SB_CONST(P_sparse_none_feat_ind, dummy_sentinal, 1);
  SB_CONST(P_sparse_none_feat_ind0, SENTINAL16, 1);
  SB_CONST(P_sparse_none_feat_ind1, SENTINAL16, 1);
  SB_CONST(P_sparse_none_feat_ind1, SENTINAL16, 1);
  SB_CONST(P_sparse_none_feat_ind1, SENTINAL16, 1);


  SB_CONST(P_sparse_none_node_ind, node_sentinal, 1);
  
  // TODO: y is now 32-bits only (page fault here)
  // TODO: can we have labels 32-bits?
  // itype, dtype, mult, offset
  // SB_CONFIG_INDIRECT1(T64, T64, 16, 8);
  SB_CONFIG_INDIRECT1(T16, T32, 2, 1);
  // SB_INDIRECT(P_IND_1, &labels[0][0], n, P_sparse_none_label);
  SB_INDIRECT(P_IND_1, &y[0], n, P_sparse_none_label);

  SB_CONST(P_sparse_none_local_offset, local_offset, n);

  // SB_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
  SB_CONFIG_ATOMIC_SCR_OP(T16, T32, T32);
  SB_ATOMIC_SCR_OP(P_sparse_none_C, P_sparse_none_D, offset, 2*n*4, 0);
  uint64_t y;
  SB_RECV(P_sparse_none_all_done, y);
  SB_RESET();
  SB_WAIT_ALL();

















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

    // SB_DMA_SCRATCH_LOAD(&fixed_inst_feat[0][0], 2, 2, N*8, getLinearAddr(0));
    // SB_WAIT_ALL();
 
    begin_roi();
    SB_CONFIG(sparse_none_config, sparse_none_size);
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

    // cur_node = node;
  }
}

struct TNode* root;

int main() {

  init_data();

  string str(file);

  FILE* train_file = fopen(str.c_str(), "r");

  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  cout << "Start reading train file!\n";
  
  int inst_id1=0;
  while(fgets(lineToRead, 5000, train_file) != NULL) {
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	char ignore;
	float x;
	int ind;

	iss >> x;
	y[inst_id1] = DOUBLE_TO_FIX(x);

    // TODO: how to make sure that ind is less than 4? some preprocessing on
    // the dataset
	while(iss >> ind) {
	  iss >> ignore >> x;
	  feat_ind[inst_id1].push_back(ind);
	  feat_val[inst_id1].push_back(DOUBLE_TO_FIX(x));
	}
    inst_id1++;
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
