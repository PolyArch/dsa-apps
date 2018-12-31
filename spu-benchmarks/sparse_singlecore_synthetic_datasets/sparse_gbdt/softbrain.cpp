#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "sparse_none.dfg.h"
#include "../../common/include/ss_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>
#define iters 63
// #define ITYPE double
#define ITYPE uint64_t
//  #define ITYPE uint16_t

#define dummy_addr (640 | (640 << 16) | ((640 & 0xFFFFFFFFFFFFFFFF) << 32) | ((640 & 0xFFFFFFFFFFFFFFFF) << 48))

#define dummy_sentinal (SENTINAL16 | (SENTINAL16 << 16) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48))

#define node_sentinal (SENTINAL16 | (0 << 16) | ((0 & 0xFFFFFFFFFFFFFFFF) << 32) | ((0 & 0xFFFFFFFFFFFFFFFF) << 48))

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
    ITYPE data_val[N][int(M/Mt)]; // load directly from your data
    ITYPE data_ind[N][int(M/Mt)]; // indexes
	ITYPE labels[N][2];
    vector<ITYPE> common_inst_id; // indexes

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
	  /*
	  for(int i=0; i<n; ++i){
	    std::cout << common_inst_id[i] << "\n";
	  }
	  */
      node->child1->inst_id.resize(n);
      node->child2->inst_id.resize(n);
      cout << "Starting process for a node\n";
      int n_f = 1;
	  int data_size = (int)(N*ratio);
	  

	  // std::cout << "(earlier) total sparse data numbers belonging to this feature: " << N*ratio << "\n";

	  // maybe good to have here
	  // data_size = (data_size/4)*4;

	  uint64_t x[2*n];
	  for(uint64_t i=0; i<2*n; ++i){
		x[i]=i;
	  }
	
	  uint64_t y;

	  std::cout << "number of instances belonging to this node: " << n << "\n";
	  std::cout << "total sparse data numbers belonging to this feature: " << data_size << "\n";
	  // std::cout << &data_ind[1][M/Mt-1]-&data_ind[0][0] << "\n";

      // can parallelize across features
      begin_roi();

	  SS_CONST_SCR(0, 0, (64*8));
      SS_WAIT_SCR_WR();

      for(int j=0; j<1; ++j) {
        // 4 feat values at a time
        SS_CONFIG(sparse_none_config,sparse_none_size);
                
		// TODO: check chances of deadlock here
		SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_sparse_none_node_ind);
		// SS_DMA_READ(&data_ind[0][0], 8, 8, data_size/4, P_sparse_none_feat_ind);
		SS_DMA_READ(&data_ind[0][0], 8*(int(M/Mt)), 8, data_size, P_sparse_none_feat_ind);
		SS_DMA_READ(&data_val[0][0], 8*(int(M/Mt)), 8, data_size, P_sparse_none_feat_val);

		// dummy address value
		SS_CONST(P_sparse_none_feat_val, dummy_addr, 1);
		// SS_CONST(P_sparse_none_feat_val, dummy_addr, 10000);
		SS_CONST(P_sparse_none_feat_ind, dummy_sentinal, 1);
		SS_CONST(P_sparse_none_node_ind, node_sentinal, 1);
	
		// value to represent that the node stream has ended
		// SS_CONST(P_sparse_none_feat_ind, SENTINAL, 1);
		// SS_CONST(P_sparse_none_node_ind, SENTINAL, 1);
		// for label, I will just do indirect read: would this be better than
		// another level of index matching?
        
		SS_DMA_READ(&node->inst_id[0], 8, 8, n, P_IND_1);
        // SS_CONFIG_INDIRECT1(T64, T64, 2, 1);
        // SS_CONFIG_INDIRECT1(T64, T64, 2*8, 1*8);
        SS_CONFIG_INDIRECT1(T64, T64, 16, 8);
        SS_INDIRECT(P_IND_1, &labels[0][0], n, P_sparse_none_label);

		// SS_DMA_READ(&x[0], 8, 8, 2*n, P_sparse_none_label);
		SS_CONST(P_sparse_none_local_offset, local_offset, n);

		// we don't know how many is required!: when all the addresses are
		// dummy, I want to reset
        SS_CONFIG_ATOMIC_SCR_OP(T16, T64, T64);
        SS_ATOMIC_SCR_OP(P_sparse_none_C, P_sparse_none_D, offset, 2*n*4, 0);
		SS_RECV(P_sparse_none_all_done, y);
		SS_RESET();
		// SS_WAIT_SCR_WR();
        SS_WAIT_ALL();
      }
      // SS_WAIT_ALL();
      
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
  uint16_t temp_val[Mt];
  uint16_t temp_ind[Mt];
  union test grad;

  cout << "Start reading file!\n";
  while(fgets(lineToRead, 5000, train_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    char ignore;

    for(int i=0; i<Mt; i++){
        iss >> temp_ind[i] >> ignore >> temp_val[i];
    }
    // iss >> t.output >> ignore >> grad.output;
    // iss >> t.output  >> grad.output;
    // std::cout << t.output << " " << grad.output << "\n";

    for(int i=0; i<Mt/4; ++i){
      tree.data_val[id][i] = (temp_val[i*4] | temp_val[i*4+1] << 16 | (temp_val[i*4+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (temp_val[i*4+3] & 0xFFFFFFFFFFFFFFFF) << 48);
      tree.data_ind[id][i] = (temp_ind[i*4] | temp_ind[i*4+1] << 16 | (temp_ind[i*4+2] & 0xFFFFFFFFFFFFFFFF) << 32 | (temp_ind[i*4+3] & 0xFFFFFFFFFFFFFFFF) << 48);
    }
    // tree.data_val[id][M] = t.out;
    // tree.labels[id][0] = t.out;
    // tree.data_val[id][M+1] = grad.out;
    // tree.labels[id][1] = grad.out;
    id++;
  }

  fclose(train_file);
  cout << "Done reading file!\n";

  id=0;
  FILE* labels_file = fopen("labels.data", "r");
  while(fgets(lineToRead, 5000, labels_file) != NULL){
	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
	iss >> t.output >> grad.output;
	tree.labels[id][0] = t.out;
	tree.labels[id][1] = grad.out;
    // iss >> tree.labels[id][0] >> tree.labels[id][1];
	id++;
  };
  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
    init_hists.label_hist[i] = 0.0;
    init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(uint64_t i=0; i<M; ++i) {
    hists.push_back(init_hists);
  }
  uint16_t x[4]; // = {0, 0, 0, 0};
  FILE* inst_file = fopen("inst_id.data", "r");
  vector<uint64_t> inst_id;
  uint16_t y;
  uint64_t temp;
  int ind=0;
  // cout << "Number of input data read from file (should be equal to N): " << tree.data.size() << "\n";
  while(fgets(lineToRead, 5000, inst_file) != NULL){

	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    char ignore;
    // iss >> x[ind];
    // iss >> temp;
    iss >> y;
	std::cout << "inst_id: " << y << std::endl;
	temp = (y | (0 << 16) | ((0 & 0xFFFFFFFFFFFFFFFF) << 32) | ((0 & 0xFFFFFFFFFFFFFFFF) << 48));

    temp = y & 0xFFFFFFFFFFFFFFFF;
	// std::cout << "inst_id: " << (temp) << std::endl;
	inst_id.push_back(temp);
	tree.common_inst_id.push_back(temp);

	// std::cout << x[ind] << " : " << ind << "\n";
	// ind++;
	// if(ind==4){
	//   // std::cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
    //   temp = (x[0] | (x[1] << 16) | ((x[2] & 0xFFFFFFFFFFFFFFFF) << 32) | ((x[3] & 0xFFFFFFFFFFFFFFFF) << 48));
	//   inst_id.push_back(temp);
	//   tree.common_inst_id.push_back(temp);
	//   ind=0;
	// }
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
