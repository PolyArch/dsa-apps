#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <sstream>
#include <sys/time.h>
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
  // int id;
  vector<int> f; // after binning
  float y;
  float z;
};

struct SplitInfo {
  int split_feat_id;
  int thres;
  float entropy;
};

struct featInfo {
  // int hist[64];
  // float hess_hist[64];
  // float grad_hist[64];
  float label_hist[64];
  float count_hist[64];
};

struct TNode {
  // float hess;
  // float grad;
  // vector<instance> inst;
  vector<int> inst_id;
  struct TNode* child1;
  struct TNode* child2;
  // histograms associated with each node
  vector<featInfo> feat_hists;
  SplitInfo info; // check?
};

class DecisionTree {

  public:
  vector<TNode> tree;
  unsigned int min_children;
  int max_depth;
  // int N; // number of instances
  // int M; // number of features
  int depth = 0; // temp variable
  vector<instance> data; // load directly from your data

  void init_data() {
    min_children = 2;
    max_depth = 10;
    // take this as input
    // N = n;
    // M = m;
  }

  /*
  int getN(){
    return N;
  }

  int getM(){
    return M;
  }
  */

  TNode init_node(){
    // struct featInfo init_hists = {{0.0}, {0.0}, {0.0}};
    struct featInfo init_hists = {{0.0}, {0.0}};
    for(int i=0; i<64; ++i){
      init_hists.label_hist[i] = 0.0;
      init_hists.count_hist[i] = 0.0;
    }
    vector<featInfo> hists;
    for(int i=0; i<M; ++i) {
      hists.push_back(init_hists);
    }

    // vector<instance> data;
    vector<int> inst_id;
    struct SplitInfo init_info = {0, 0, 0.0};
  
    // TNode temp = {2.1, 3.5, data, nullptr, nullptr, hists, init_info};
    // TNode temp = {data, nullptr, nullptr, hists, init_info};
    TNode temp = {inst_id, nullptr, nullptr, hists, init_info};
    return temp;
  }

  void build_tree(struct TNode* node){
    float entr;
    float max_entr;
    int b;
    int cur_label;
    int cur_count;
    int max_label;
    TNode child1[100];
    TNode child2[100];
    
    vector<TNode*> cur_nodes;
    cur_nodes.push_back(node);
	unsigned node_id=0;
    // for(unsigned node_id=0; node_id<cur_nodes.size() && depth < max_depth; ++node_id){
	{
      max_entr = 0.0;
      node = cur_nodes[node_id];
      child1[node_id] = init_node();
      child2[node_id] = init_node();
      node->child1 = &child1[node_id];
      node->child2 = &child2[node_id];
      // node->child1->inst_id.resize(node->inst_id.size());
      // node->child2->inst_id.resize(node->inst_id.size());
      // can parallelize across features
	  begin_roi();
      for(int j=1; j<Mt; ++j) {

        // histogram building
        for(unsigned int i=0; i<node->inst_id.size(); ++i) {
           b = data[node->inst_id[i]].f[j];
           node->feat_hists[j].label_hist[b] += data[node->inst_id[i]].y;
           node->feat_hists[j].count_hist[b] += data[node->inst_id[i]].z;
           // node->feat_hists[j].count_hist[b] += 1;
        }
       
		/*
        // find cumulative sum
        for(int i=1; i<64; ++i) {
            node->feat_hists[j].label_hist[i] += node->feat_hists[j].label_hist[i-1];
            node->feat_hists[j].count_hist[i] += node->feat_hists[j].label_hist[i-1];
        }

        // reduction step
        max_label = node->feat_hists[j].label_hist[63];
        for(unsigned int thresh=0; thresh<64; ++thresh) { // threshold of 0 and 63 are not allowed
          cur_label = node->feat_hists[j].label_hist[thresh];
          cur_count = node->feat_hists[j].count_hist[thresh];
          entr = pow(cur_label,2)/(cur_count) + pow(max_label-cur_label,2)/(N-cur_count);

          if(entr>max_entr){
            max_entr = entr;
            node->info.split_feat_id = j;
            node->info.thres = thresh;
          } 
        }
		*/
      }

	  /*
      for(unsigned int i=0; i<node->inst_id.size(); ++i) {
        if(data[node->inst_id[i]].f[node->info.split_feat_id] <= node->info.thres) {
          // allot to child1
          node->child1->inst_id.push_back(i);
        } else {
          node->child2->inst_id.push_back(i);
        }
      }

      // cout << "Depth: " << depth << "\n"; 
      if(node->child1->inst_id.size() > min_children){
        cur_nodes.push_back(node->child1);
        // cout << "child1 size: " << node->child1->inst_id.size() <<  "\n";
      }
      if(node->child2->inst_id.size() > min_children){
        cur_nodes.push_back(node->child2);
        // cout << "child2 size: " << node->child2->inst_id.size() <<  "\n";
      }
      if(node->child1->inst_id.size() > min_children || node->child2->inst_id.size() > min_children){
       depth++;   
      }
	  */
   }
	end_roi();
 }
};

int main() {
  DecisionTree tree;

  // tree.init_data(65536,8);
  tree.init_data();

  // N = 3; M = 2;
  vector<float> input;
  vector<int> binned_input; // binned: I should do binning in CGRA? No, it is required only once
  for(int i=0; i<M; ++i) {
    input.push_back(0.0);
    binned_input.push_back(0.0);
  }
  float output1; float output2;
  int id=0;
  // instance temp = {1, binned_input, 100};
  instance temp = {binned_input, 100, 100};
  FILE* train_file = fopen("input.data", "r");
  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  printf("Started reading file!\n");
 
  while(fgets(lineToRead, 500000, train_file) != NULL) {
    if(*lineToRead==',') { continue; }

	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());

	for(int i=0; i<M; i++){
	  iss >> binned_input[i];
	}
	iss >> output1;
	iss >> output2;

    tree.data.push_back(temp);
    id++;
  }

  fclose(train_file);
  printf("Done reading file!\n");

  // struct featInfo init_hists = {{0.0}, {0.0}, {0.0}};
  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
      init_hists.label_hist[i] = 0.0;
      init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(int i=0; i<M; ++i) {
    hists.push_back(init_hists);
  }
  vector<int> inst_id;
  /*
  for(unsigned i=0; i<tree.data.size(); ++i) {
    inst_id.push_back(i);
  }
  */

  FILE* id_file = fopen("inst_id.data", "r");

  printf("Started reading inst id file!\n");
  int l=0;
 
  while(fgets(lineToRead, 500000, id_file) != NULL) {
	std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
	iss >> l;
	inst_id.push_back(l);

  } 




  struct SplitInfo init_info = {0, 0, 0.0};
  // struct TNode node = {2.3, 3.5, data, nullptr, nullptr, hists, init_info};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}

