#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "test.dfg.h"
#include "none.dfg.h"
#include "final_redn.dfg.h"
#include "map.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>
// #define Ni 10000
#define Ni 100
#define Mi 8
#define iters 63

using namespace std;

struct instance {
  // vector<uint64_t> f; // after binning
  uint64_t f[Mi+1]; // after binning
  // double y;
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
    uint64_t N; // number of instances
    uint64_t M; // number of features
    uint64_t depth = 0; // temp variable
    vector<instance> data; // load directly from your data

  void init_data(uint64_t n, uint64_t m) {
    min_children = 2;
    max_depth = 10;
    N = n;
    M = m;
  }

  uint64_t getN(){
    return N;
  }

  uint64_t getM(){
    return M;
  }

  TNode init_node(){
    struct featInfo init_hists = {{0.0}, {0.0}};
    for(int i=0; i<64; ++i){
      init_hists.label_hist[i] = 0.0;
      init_hists.count_hist[i] = 0.0;
    }
    vector<featInfo> hists;
    for(uint64_t i=0; i<getM(); ++i) {
      hists.push_back(init_hists);
    }

    vector<uint64_t> inst_id;
    struct SplitInfo init_info = {0, 0, 0.0};
  
    TNode temp = {inst_id, nullptr, nullptr, hists, init_info};
    return temp;
  }

  void build_tree(struct TNode* node){
    double max_entr;
    uint64_t n;
    TNode child1[100];
    TNode child2[100];
    
    vector<TNode*> cur_nodes;
    cur_nodes.push_back(node);
    for(unsigned node_id=0; node_id<cur_nodes.size() && depth < max_depth; ++node_id){
      max_entr = 0.0;
      node = cur_nodes[node_id];
      child1[node_id] = init_node();
      child2[node_id] = init_node();
      node->child1 = &child1[node_id];
      node->child2 = &child2[node_id];
      n = node->inst_id.size();

      cout << "Starting process for a node\n";
      begin_roi();
      // can parallelize across features
      for(uint64_t j=1; j<M; ++j) {

        SB_CONFIG(none_config,none_size);

        SB_CONST_SCR(0, 0, (64*2));
        SB_WAIT_SCR_WR();
        
        SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
        SB_CONFIG_INDIRECT1(T64, T64, Mi, Mi-j);
        SB_INDIRECT(P_IND_1, &data[0].f[j], n, P_none_A);

        SB_CONST(P_none_const, 1, n);
        SB_ATOMIC_SCR_OP(P_none_C, P_none_D, 0, 2*n, 0);
        SB_WAIT_SCR_WR();
        SB_WAIT_ALL();

        SB_CONFIG(test_config,test_size);

        for(int i=0; i<2; ++i) { 
          SB_SCRATCH_READ(64*i,64*8,P_test_P);
          SB_CONST(P_test_Q, 0, 64);
          SB_SCR_WRITE(P_test_T,8*64,64*i);
          SB_WAIT_SCR_WR();
        }

        // reduction step
        SB_REPEAT_PORT(iters);
        SB_SCRATCH_READ(63, 8, P_test_max_label);
        SB_REPEAT_PORT(iters);
        SB_SCRATCH_READ(64*2 - 1, 8, P_test_max_count);
        SB_SCRATCH_READ(0, 8*iters, P_test_label);
        SB_SCRATCH_READ(64, 8*iters, P_test_count);

        SB_CONST(P_test_const1, 1, iters);
        SB_CONST(P_test_const2, 0, iters);
        // SB_CONST(P_test_const2, 0, iters-1);
        // SB_CONST(P_test_const2, 1, 1);
        // SB_2D_CONST(P_test_const2, 0, iters-1, 0, 1, 1);
       
        SB_SCR_WRITE(P_test_final_entr, 8*1, 64*2+j);
        SB_SCR_WRITE(P_test_split_thres, 8*1, 64*2+1+j);

        SB_WAIT_SCR_WR();
        SB_WAIT_ALL();
      }
      SB_WAIT_ALL();
      // cout << "Node 1 done\n";
      // final reduction step to find the split
      SB_CONFIG(final_redn_config, final_redn_size);
      SB_SCR_PORT_STREAM(64*2, 8*2, 8, (M-1), P_final_redn_cur_entr);
      SB_CONST(P_final_redn_const1, 1, M-1);
      SB_CONST(P_final_redn_const2, 0, M-1);
      // let's write it in dma
      SB_SCR_WRITE(P_final_redn_split_feat_id, 8*1, 64*2+M+1);
      SB_WAIT_SCR_WR();
      // could have stored directly to dram
      SB_SCRATCH_DMA_STORE(64*2+M+1, 8, 8, 1, &node->info.split_feat_id);
      SB_WAIT_ALL();
      SB_SCRATCH_DMA_STORE(64*2+node->info.split_feat_id*2+1, 8, 8, 1, &node->info.thres);
      SB_WAIT_ALL();

      // for min: it should find b/w two addresses (val should be addr)
      
      // this has to be repeated multiple times--better to keep in SCR: RD/WR
      // TODO: NEED TO MALLOC
      // n = 10;
      uint64_t z[n];
      for(int i=0; i<n; i++){
        z[i] = i;
      }
      SB_CONFIG(map_config, map_size);
      // SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_DOUB0);
      SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
      SB_CONFIG_INDIRECT(T64, T64, Mi);
      // SB_INDIRECT(P_IND_DOUB0, &data[0].f[node->info.split_feat_id], n, P_map_feat_val); 
      SB_INDIRECT(P_IND_1, &data[0].f[node->info.split_feat_id], n, P_map_feat_val); 
      SB_CONST(P_map_split_thres, node->info.thres, n);
      // SB_CONST(P_map_child1_offset, &node->child1->inst_id[0], n); // we want it to yield final address
      SB_CONST(P_map_child1_offset,1, n); // we want it to yield final address
      // SB_CONST(P_map_child2_offset, &node->child2->inst_id[0], n);
      SB_CONST(P_map_child2_offset, 2, n);
      SB_DMA_WRITE(P_map_offset, 8, 8, n, &z[0]);
      // SB_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_2);
      // SB_CONFIG_INDIRECT(T64, T64, 1);
      // SB_INDIRECT_WR(P_IND_DOUB1, 0, n, P_map_offset);
      // SB_INDIRECT_WR(P_IND_2, 0, n, P_map_offset);
      end_roi();
      sb_stats();


      /*
      for(unsigned int i=0; i<node->inst_id.size(); ++i) {
        if(data[node->inst_id[i]].f[node->info.split_feat_id] <= node->info.thres) {
          // allot to child1
          node->child1->inst_id.push_back(i);
        } else {
          node->child2->inst_id.push_back(i);
        }
      }

      if(node->child1->inst_id.size() > min_children){
        cur_nodes.push_back(node->child1);
      }
      if(node->child2->inst_id.size() > min_children){
        cur_nodes.push_back(node->child2);
      }
     
    */
    }
  }
};

int main() {
  DecisionTree tree;

  // tree.init_data(65536,8);
  // tree.init_data(10000,8);
  tree.init_data(Ni,Mi);

  // N = 3; M = 2;
  vector<double> input;
  instance temp;
  for(uint64_t i=0; i<tree.getM(); ++i) {
    input.push_back(0.0);
    temp.f[0] = 0;
  }
  temp.f[Mi] = 0;
  double output; int id=0;
  // FILE* train_file = fopen("train_data_short.csv", "r");
  FILE* train_file = fopen("test_data.csv", "r");
  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  /* float min_val[tree.getM()];
  for(int i=0; i<tree.getM(); ++i){
      min_val[i] = 0;
  }
  */
  cout << "Start reading file!\n";
  while(fgets(lineToRead, 500000, train_file) != NULL) {
    if(*lineToRead==',') { continue; }
    sscanf(lineToRead, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &input[0], &input[1], &input[2], &input[3], &input[4], &input[5], &input[6], &input[7], &output);
    for(uint64_t i=0; i<tree.getM(); ++i){
      if(!id){
        temp.f[i] =  (uint64_t)input[i]%64;
      } else{
        temp.f[i] = 0;
      }
    }
    if(!id) {
      temp.f[Mi] = output*std::rand()/(RAND_MAX + 1u)/1000;
    } else {
      temp.f[Mi] = 0;
    }
    tree.data.push_back(temp);
    id++;
  }

  fclose(train_file);
  cout << "Done reading file!\n";

  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
    init_hists.label_hist[i] = 0.0;
    init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(uint64_t i=0; i<tree.getM(); ++i) {
    hists.push_back(init_hists);
  }
  vector<uint64_t> inst_id;
  for(unsigned i=0; i<tree.data.size(); ++i) {
    inst_id.push_back(i);
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}

