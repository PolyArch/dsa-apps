#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "test.dfg.h"
#include "gbdt.dfg.h"
#include "final_redn.dfg.h"
#include "map.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <sstream>
#define iters 63
// #define ITYPE double
#define ITYPE uint64_t
//  #define ITYPE uint16_t

using namespace std;

struct instance {
  ITYPE f[M+2]; // after binning
};

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

class DecisionTree {

  public:
    uint64_t min_children;
    uint64_t max_depth;
    uint64_t depth = 0; // temp variable
    // vector<instance> data; // load directly from your data
    // instance data[N]; // load directly from your data
	// TODO: this should be different so that y and grad are far away
    ITYPE data[N][M+2]; // load directly from your data

	// Data structures to read from file
	float inst_feat[N][M];
	uint32_t inst_label[N];

	// Dynamic data structures
	uint16_t fixed_inst_feat[N][M];
	uint32_t hess[N];
	uint32_t grad[N];

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

/*
  void feature_binning() {
	// fixed_inst_feat[n][i] = DOUBLE_TO_FIX(tree.inst_feat[n][i]);


  }
  */

  void build_tree(struct TNode* node){
    // cout << "Came here to build tree" << "\n";
    double max_entr;
    uint64_t n;
    TNode child1[100];
    TNode child2[100];

    vector<TNode*> cur_nodes;
    cur_nodes.push_back(node);
    // cout << "Current nodes in queue are: " << cur_nodes.size() << " depth is: " << depth << "\n";
    uint64_t offset = (k*0 | k*2 << 16 | (k*4 & 0xFFFFFFFFFFFFFFFF) << 32 | (k*6 & 0xFFFFFFFFFFFFFFFF) << 48);
    uint64_t local_offset1 = (k | k << 16 | (k & 0xFFFFFFFFFFFFFFFF) << 32 | (k & 0xFFFFFFFFFFFFFFFF) << 48);
    uint64_t local_offset2 = ((2*k) | (2*k) << 16 | ((2*k) & 0xFFFFFFFFFFFFFFFF) << 32 | ((2*k) & 0xFFFFFFFFFFFFFFFF) << 48);

	uint64_t fused_const = (1 | 1 << 16 | (1 & 0xFFFFFFFFFFFFFFFF) << 32 | (1 & 0xFFFFFFFFFFFFFFFF) << 48);
    
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
      int n_f = 1;


      begin_roi();

      int j=0;

	  // TODO: change indirect type also
	  SB_CONFIG(gbdt_config,gbdt_size);

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
      SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, offset, n*4*3, 0);
      SB_WAIT_SCR_WR();
      SB_WAIT_ALL();

      end_roi();
      sb_stats();
    }
  }
};

int main() {
  DecisionTree tree;

  tree.init_data();

  FILE* train_file = fopen("binned_small_mslr.train", "r");
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

	iss >> tree.inst_label[n];

    for(int i=0; i<M; i++){
	  // iss >> index >> ignore >> tree.inst_feat[n][i];
      // iss >> index >> ignore >> tree.fixed_inst_feat[n][i];
      iss >> tree.fixed_inst_feat[n][i];
	  // tree.fixed_inst_feat[n][i] = DOUBLE_TO_FIX(tree.inst_feat[n][i]);
	}
	n++;
  }

  fclose(train_file);
  cout << "Done reading file!\n";

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
	tree.hess[i] = DOUBLE_TO_FIX(float(i*0.2));
	tree.grad[i] = DOUBLE_TO_FIX(float(i*0.3));
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}
