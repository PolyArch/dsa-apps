#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "test.dfg.h"
#include "stall_none.dfg.h"
#include "final_redn.dfg.h"
#include "map.dfg.h"
#include "../../common/include/ss_insts.h"
#include "../../common/include/sim_timing.h"
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
    uint64_t local_offset = (k | k << 16 | (k & 0xFFFFFFFFFFFFFFFF) << 32 | (k & 0xFFFFFFFFFFFFFFFF) << 48);
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



      int j=0;
	  // for instruction stalls
	  // 64*8*64 bits / 512 = 64 cycles
	  SS_CONST_SCR(0, 0, (64*8));
      SS_WAIT_SCR_RD(); // should prevent write because my atomic scr is both rd and wr but in wr controller?
      SS_WAIT_SCR_WR();

	  SS_CONFIG(stall_none_config,stall_none_size);
      SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_TRIP0);
      SS_CONFIG_INDIRECT(T64, T64, M+2);
      SS_INDIRECT(P_IND_TRIP0, &data[0][j], n, P_stall_none_A);
      
      SS_CONFIG_INDIRECT(T64, T64, M+2);
      SS_INDIRECT(P_IND_TRIP1, &data[0][M], n, P_stall_none_label);

      SS_CONFIG_INDIRECT(T64, T64, M+2);
      SS_INDIRECT(P_IND_TRIP2, &data[0][M+1], n, P_stall_none_const);

      SS_CONST(P_stall_none_local_offset, local_offset, n);

      SS_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
      SS_ATOMIC_SCR_OP(P_stall_none_C, P_stall_none_D, offset, 2*n*4, 0);
      SS_WAIT_SCR_WR();
      SS_WAIT_ALL();



      // can parallelize across features
      // for(uint64_t j=0; j<M; ++j) {
      begin_roi();
	
      // for(int j=0; j<1; ++j) {
      // for(int j=0; j<1; ++j) {
	  {
        // 4 feat values at a time
		/*

        SS_CONST_SCR(0, 0, (64*8));
        SS_WAIT_SCR_RD(); // should prevent write because my atomic scr is both rd and wr but in wr controller?
        SS_WAIT_SCR_WR();



        SS_CONFIG(stall_none_config,stall_none_size);
                

        // SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
        // SS_CONFIG_INDIRECT2(T64, T64, M+2, M-j, M+1-j);
        // SS_INDIRECT(P_IND_1, &data[0][j], n, P_stall_none_A);
        // SS_INDIRECT(P_IND_1, &data[0][j], n, P_stall_none_label);
        // SS_INDIRECT(P_IND_1, &data[0][j], n, P_stall_none_const);
        // TODO: I can use offset list here now
        SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_TRIP0);
        SS_CONFIG_INDIRECT(T64, T64, M+2);
        SS_INDIRECT(P_IND_TRIP0, &data[0][j], n, P_stall_none_A);
        
        SS_CONFIG_INDIRECT(T64, T64, M+2);
        SS_INDIRECT(P_IND_TRIP1, &data[0][M], n, P_stall_none_label);

        SS_CONFIG_INDIRECT(T64, T64, M+2);
        SS_INDIRECT(P_IND_TRIP2, &data[0][M+1], n, P_stall_none_const);

        SS_CONST(P_stall_none_local_offset, local_offset, n);

        SS_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
        SS_ATOMIC_SCR_OP(P_stall_none_C, P_stall_none_D, offset, 2*n*4, 0);
        SS_WAIT_SCR_WR();
        SS_WAIT_ALL();




        // SS_CONFIG(none_config,none_size);
        //         
        // SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_TRIP0);
        // SS_CONFIG_INDIRECT(T64, T64, M+2);
        // SS_INDIRECT(P_IND_TRIP0, &data[0][j], n, P_none_A);
        // 
        // SS_CONFIG_INDIRECT(T64, T64, M+2);
        // SS_REPEAT_PORT(4);
        // SS_INDIRECT(P_IND_TRIP1, &data[0][M], n, P_none_label);

        // SS_CONFIG_INDIRECT(T64, T64, M+2);
        // SS_REPEAT_PORT(4);
        // SS_INDIRECT(P_IND_TRIP2, &data[0][M+1], n, P_none_const);

        // SS_CONST(P_none_local_offset, local_offset, n);

        // SS_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
        // SS_ATOMIC_SCR_OP(P_none_C, P_none_D, offset, 2*n*4, 0);
        // SS_WAIT_ALL();

        SS_CONFIG(test_config,test_size);

        for(int f=0; f<4; ++f){
          for(int i=0; i<2; ++i) { 
            SS_SCRATCH_READ(64*i+128*f, 64*8, P_test_P);
            SS_CONST(P_test_Q, 0, 64);
            SS_SCR_WRITE(P_test_T, 8*64, 64*i+128*f);
            SS_WAIT_SCR_WR();
          }
          SS_WAIT_ALL();
        }
        
        SS_CONFIG(test_config,test_size);
        // for(int f=0; f<4; ++f){
        // only this thing is left----------------
        for(int f=0; f<1; ++f){
          // reduction step: this can be vectorized if we are ready to reconfigure
          SS_REPEAT_PORT(iters);
          SS_SCRATCH_READ(63+128*f, 8, P_test_max_label);
          SS_REPEAT_PORT(iters);
          SS_SCRATCH_READ(64*2+128*f - 1, 8, P_test_max_count);
        
          SS_SCRATCH_READ(128*f, 8*iters, P_test_label);
          SS_SCRATCH_READ(64+128*f, 8*iters, P_test_count);
          SS_CONST(P_test_const1, 1, iters*n_f);
          SS_CONST(P_test_const2, 0, iters*n_f);
       
          SS_SCR_WRITE(P_test_final_entr, 8*1, 128*f+64*2+j);
          SS_SCR_WRITE(P_test_split_thres, 8*1, 128*f+64*2+1+j);

          SS_WAIT_SCR_WR();
          SS_WAIT_ALL();
        }
		*/
      }
      // SS_WAIT_ALL();
      // end_roi();
      // sb_stats();
	  /*
      int m=8;
      SS_CONFIG(final_redn_config, final_redn_size);
      SS_SCR_PORT_STREAM(64*2, 8*2, 8, (m-1), P_final_redn_cur_entr);
      SS_CONST(P_final_redn_const1, 1, m-1);
      SS_CONST(P_final_redn_const2, 0, m-1);
      // let's write it in dma
      SS_SCR_WRITE(P_final_redn_split_feat_id, 8*1, 64*2+m+1);
      SS_WAIT_SCR_WR();
      SS_WAIT_SCR_RD();
      // could have stored directly to dram
      SS_SCRATCH_DMA_STORE(64*2+m+1, 8, 8, 1, &node->info.split_feat_id);
      SS_WAIT_ALL();
      SS_SCRATCH_DMA_STORE(64*2+node->info.split_feat_id*2+1, 8, 8, 1, &node->info.thres);
      SS_WAIT_ALL();

      double split_thres = node->info.thres; 
      int feat_id = node->info.split_feat_id;

      // ideally should be in different scratchpad
      SS_CONFIG(map_config, map_size);

      // extra read of id's: do something
      SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
      SS_CONFIG_INDIRECT(T64, T64, M+2);
      SS_INDIRECT(P_IND_1, &data[0][feat_id], n, P_map_feat_val);
      SS_CONST(P_map_split_thres, split_thres, n);
      SS_CONST(P_map_child1_offset, 0, n);
      SS_CONST(P_map_child2_offset, 64, n);
      SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_map_node_id);

      SS_CONFIG_ATOMIC_SCR_OP(T64, T64, T64);
      // SS_ATOMIC_SCR_OP(P_map_offset, P_IND_DOUB1, 0, n, 3); // I want just update
      SS_ATOMIC_SCR_OP(P_map_offset, P_map_id, 0, n, 3); // I want just update
 
      // indirect wr to only scr possible: not allowed to use output port as
      // indirect port? (weird): should i remove that condition?
      // SS_INDIRECT_WR(P_map_offset, 0, n, P_map_id); // it is not mapping correct value as the output...need to see
      SS_WAIT_ALL();
	  */


      end_roi();
      sb_stats();

    }
    
  }
};

union test{
    double output;
    uint64_t out;
};

int main() {
  DecisionTree tree;

  tree.init_data();

  // double output; 
  union test t;
  int id=0;
  
  FILE* train_file = fopen("input.data", "r");
  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }
  uint16_t temp2[Mt];
  union test grad;

  cout << "Start reading file!\n";
  while(fgets(lineToRead, 5000, train_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    char ignore;

    for(int i=0; i<Mt; i++){
        iss >> temp2[i];
    }
    // iss >> t.output >> ignore >> grad.output;
    iss >> t.output  >> grad.output;
    // std::cout << t.output << " " << grad.output << "\n";

    tree.data[id][0] = (temp2[0] | temp2[1] << 16 | (temp2[2] & 0xFFFFFFFFFFFFFFFF) << 32 | (temp2[3] & 0xFFFFFFFFFFFFFFFF) << 48);
	/*
    for(int i=0; i<Mt/4; ++i){
      tree.data[id][i] = (temp2[i*4] | temp2[i*4+1] << 16 | (temp2[i*4+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (temp2[i*4+3] & 0xFFFFFFFFFFFFFFFF) << 48);
    }
	*/
    tree.data[id][M] = t.out;
    tree.data[id][M+1] = grad.out;
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
  for(uint64_t i=0; i<M; ++i) {
    hists.push_back(init_hists);
  }
  uint64_t x = 0;
  FILE* inst_file = fopen("inst_id.data", "r");
  vector<uint64_t> inst_id;
  // cout << "Number of input data read from file (should be equal to N): " << tree.data.size() << "\n";
  while(fgets(lineToRead, 5000, inst_file) != NULL){
    sscanf(lineToRead, "%ld ", &x);
    inst_id.push_back(x);
  };
  /*
  for(unsigned i=0; i<n*ratio; ++i) {
    inst_id.push_back(i);
  }
  */

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}
