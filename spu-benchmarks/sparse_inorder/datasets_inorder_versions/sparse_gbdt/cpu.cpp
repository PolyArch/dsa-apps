#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include <sstream>
#include <sys/time.h>
#define iters 63
// #define ITYPE double
#define ITYPE uint64_t
//  #define ITYPE uint16_t

#define dummy_addr (640 | (640 << 16) | ((640 & 0xFFFFFFFFFFFFFFFF) << 32) | ((640 & 0xFFFFFFFFFFFFFFFF) << 48))

#define dummy_sentinal (SENTINAL16 | (SENTINAL16 << 16) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 32) | ((SENTINAL16 & 0xFFFFFFFFFFFFFFFF) << 48))

#define node_sentinal (SENTINAL16 | (0 << 16) | ((0 & 0xFFFFFFFFFFFFFFFF) << 32) | ((0 & 0xFFFFFFFFFFFFFFFF) << 48))

using namespace std;

static uint64_t ticks;

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static void begin_roi() {

  ticks=rdtsc();

}


static void end_roi()   {

  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);

}
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
  vector<uint16_t> inst_id;
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
    // uint16_t data_val[N][M]; // load directly from your data
    // uint16_t data_ind[N][M]; // indexes

    vector<uint16_t> data_val[M];
    vector<uint16_t> data_ind[M];

	double labels[N][2];
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

    vector<uint16_t> inst_id;
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
      // cout << "Starting process for a node\n";

	  int ptr1=0, end1=0, ptr2=0, end2=0;

	  // std::cout << "number of instances belonging to this node: " << n << "\n";
	  // std::cout << "total sparse data numbers belonging to this feature: " << data_size << "\n";

      // can parallelize across features
      // begin_roi();
      for(int j=0; j<4; ++j) {
		// For 1 feature
		ptr1=0;
		ptr2=0;
		end1 = node->inst_id.size()-1;
		end2 = data_val[j].size()-1;
		// cout << "ptr1: " << ptr1 << " end1: " << end1 << " ptr2: " << ptr2 << " end2: " << end2 << "\n";
 
		while(ptr1 <= end1 && ptr2 <= end2){
		  int node_id = node->inst_id[ptr1];
		  if(node_id == data_ind[j][ptr2]){
			node->feat_hists[j].label_hist[data_val[j][ptr2]] += labels[node_id][0];
			node->feat_hists[j].count_hist[data_val[j][ptr2]] += labels[node_id][0];
			ptr1++; ptr2++;
		  } else {
			if(node_id < data_ind[j][ptr2])
			  ptr1++;
			else
			  ptr2++;
		  }
        }
      }
      // end_roi();
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
  
  // FILE* train_file = fopen("input.data", "r");

  string str(file);

  FILE* train_file = fopen(str.c_str(), "r");


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
    int x, ind;

    iss >> tree.labels[id][0];
    tree.labels[id][1]=1;

    while(iss >> ind) {
      iss >> ignore >> x;
      tree.data_ind[ind].push_back(id);
      tree.data_val[ind].push_back(x);
    }
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
  uint16_t x[4]; // = {0, 0, 0, 0};
  // FILE* inst_file = fopen("inst_id.data", "r");
  vector<uint16_t> inst_id;
  for(int i=0; i<N; ++i) inst_id.push_back(i);
  /*
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
	inst_id.push_back(y);
  };
  */
  /*
  for(unsigned i=0; i<n*ratio; ++i) {
    inst_id.push_back(i);
  }
  */

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);
  begin_roi();
  tree.build_tree(root);
  end_roi();

  return 0;
}
