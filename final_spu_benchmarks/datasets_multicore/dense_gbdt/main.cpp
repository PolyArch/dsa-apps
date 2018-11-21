#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "local_redn.dfg.h"
#include "test.dfg.h"
#include "gbdt.dfg.h"
#include "final_redn.dfg.h"
#include "map.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <sstream>
#define iters 63
#define k 64

using namespace std;

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
    
	// Data structures to read from file
	float inst_feat[N][M];
	uint32_t inst_label[N];

	// Dynamic data structures
	uint16_t fixed_inst_feat[N][M];
	uint32_t hess[N];
	uint32_t grad[N];
	uint32_t fm[k];

    // Barrier variable
    pthread_barrier_t barr;


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



  void local_reduction2() {
    // calculate the error from the three histograms for all 4 features
    // all 16-bit values (16*12=192 bits)

    SB_CONFIG(local_redn_config, local_redn_size);

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

	// find min of 4 (get on control core only)
	SB_SCRATCH_READ(k*4*12, 4*4, P_local_redn_cur_err);
    SB_CONST(P_local_redn_prev_err, 100000, 1);
    SB_RECURRENCE(P_local_redn_reduced_error, P_local_redn_prev_err, 3);
	// TODO: this should be remote scr_wr for hierarchical reduction
	SB_SCR_WRITE(P_local_redn_final_error, 4, k*4*12 + 4*4);
	SB_WAIT_ALL();
  }

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
    // SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, offset, n*4*3, 0);
	// FIXME: CHECKME: I want it to be 16-bits -- should followe address datatype
    SB_ATOMIC_SCR_OP(P_gbdt_C, P_gbdt_D, feature_offset, n*4*3, 0);
    SB_WAIT_SCR_WR();
    SB_WAIT_ALL();
 }

  void hierarchical_reduction() {
	// everybody should send to core id 0 (sequence doesn't matter)
	// portA: thres
	// portB: feat_id
	// portC: min_err
	


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
    build_histogram(tid, cur_node);
    local_reduction2();
	hierarchical_reduction();
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


	  begin_roi();
	  // build_histogram(0, node);
	  cout << "Done with histogram building\n";

	  // local_reduction1();
      local_reduction2();
	  cout << "Done with local reduction\n";
	  // hierarchical reduction
	  // broadcast of the value
	  // mapping
 
      end_roi();
      sb_stats();

/*
      cur_node = node;
      assert(NUM_THREADS<C);
      
      // Barrier initialization
      if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
      {
        printf("Could not create a barrier\n");
        return -1;
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
          return 0;
        }
      }
      
      for(int i = 0; i < NUM_THREADS; ++i) {
        if(pthread_join(threads[i], NULL)) {
      	printf("Could not join thread %d\n", i);
          return -1;
        }
      }
	  */
    }
  }
};

DecisionTree tree;
struct TNode* root;

int main() {

  tree.init_data();

  string str(file);

  // FILE* train_file = fopen("datasets/binned_small_mslr.train", "r");
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

  for(int i=0; i<k; ++i){
	tree.fm[i]=i;
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
	tree.hess[i] = DOUBLE_TO_FIX(float(i*0.2));
	tree.grad[i] = DOUBLE_TO_FIX(float(i*0.3));
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  root = &node;

  tree.build_tree(root);
  return 0;
}

  /*
  void local_reduction1() {
    // calculate the error from the three histograms for all 4 features
    // all 16-bit values (16*12=192 bits)

    // PART 1: cumulative sum of the histo values
    SB_CONFIG(test_config,test_size);

    for(int f=0; f<4; ++f){
      for(int i=0; i<2; ++i) {
        SB_SCRATCH_READ(64*i+128*f, 64*8, P_test_P);
        SB_CONST(P_test_Q, 0, 64);
        SB_SCR_WRITE(P_test_T, 8*64, 64*i+128*f);
        SB_WAIT_SCR_WR();
      }
      SB_WAIT_ALL();
    }

    // PART 2: calculate error and find minimum
    for(int f=0; f<4; ++f){
      // reduction step: this can be vectorized if we are ready to reconfigure
      SB_REPEAT_PORT(iters);
      SB_SCRATCH_READ(63+128*f, 8, P_test_max_label);
      SB_REPEAT_PORT(iters);
      SB_SCRATCH_READ(64*2+128*f - 1, 8, P_test_max_count);

      SB_SCRATCH_READ(128*f, 8*iters, P_test_label);
      SB_SCRATCH_READ(64+128*f, 8*iters, P_test_count);
      SB_CONST(P_test_const1, 1, iters*n_f);
      SB_CONST(P_test_const2, 0, iters*n_f);

      SB_SCR_WRITE(P_test_final_entr, 8*1, 128*f+64*2+j);
      SB_SCR_WRITE(P_test_split_thres, 8*1, 128*f+64*2+1+j);

      SB_WAIT_SCR_WR();
      SB_WAIT_ALL();
    }
  }
  */
