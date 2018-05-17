#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
using namespace std;

struct instance {
  // int id;
  vector<int> f; // after binning
  float y;
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
  int N; // number of instances
  int M; // number of features
  int depth = 0; // temp variable
  vector<instance> data; // load directly from your data

  void init_data(int n, int m) {
    min_children = 2;
    max_depth = 10;
    // take this as input
    N = n;
    M = m;
  }

  int getN(){
    return N;
  }

  int getM(){
    return M;
  }

  TNode init_node(){
    // struct featInfo init_hists = {{0.0}, {0.0}, {0.0}};
    struct featInfo init_hists = {{0.0}, {0.0}};
    for(int i=0; i<64; ++i){
      init_hists.label_hist[i] = 0.0;
      init_hists.count_hist[i] = 0.0;
    }
    vector<featInfo> hists;
    for(int i=0; i<getM(); ++i) {
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
    for(unsigned node_id=0; node_id<cur_nodes.size() && depth < max_depth; ++node_id){
      max_entr = 0.0;
      node = cur_nodes[node_id];
      child1[node_id] = init_node();
      child2[node_id] = init_node();
      node->child1 = &child1[node_id];
      node->child2 = &child2[node_id];

      // can parallelize across features
      for(int j=1; j<M; ++j) {

        // histogram building
        for(unsigned int i=0; i<node->inst_id.size(); ++i) {
           b = data[node->inst_id[i]].f[j];
           node->feat_hists[j].label_hist[b] += data[node->inst_id[i]].y;
           node->feat_hists[j].count_hist[b] += 1;
        }
        
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
      }

      for(unsigned int i=0; i<node->inst_id.size(); ++i) {
        if(data[node->inst_id[i]].f[node->info.split_feat_id] <= node->info.thres) {
          // allot to child1
          node->child1->inst_id.push_back(i);
        } else {
          node->child2->inst_id.push_back(i);
        }
      }

      cout << "Depth: " << depth << "\n"; 
      if(node->child1->inst_id.size() > min_children){
        cur_nodes.push_back(node->child1);
        cout << "child1 size: " << node->child1->inst_id.size() <<  "\n";
      }
      if(node->child2->inst_id.size() > min_children){
        cur_nodes.push_back(node->child2);
        cout << "child2 size: " << node->child2->inst_id.size() <<  "\n";
      }
      if(node->child1->inst_id.size() > min_children || node->child2->inst_id.size() > min_children){
       depth++;   
      }
   }
 }
};

int main() {
  DecisionTree tree;

  // tree.init_data(65536,8);
  tree.init_data(10000,8);

  // N = 3; M = 2;
  vector<float> input;
  vector<int> binned_input; // binned: I should do binning in CGRA? No, it is required only once
  for(int i=0; i<tree.getM(); ++i) {
    input.push_back(0.0);
    binned_input.push_back(0.0);
  }
  float output; int id=0;
  // instance temp = {1, binned_input, 100};
  instance temp = {binned_input, 100};
  FILE* train_file = fopen("train_data_short.csv", "r");
  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  /* float min_val[tree.getM()];
  for(int i=0; i<tree.getM(); ++i){
      min_val[i] = 0;
  }
  */
  while(fgets(lineToRead, 500000, train_file) != NULL) {
      if(*lineToRead==',') { continue; }
      sscanf(lineToRead, "%f,%f,%f,%f,%f,%f,%f,%f,%f", &input[0], &input[1], &input[2], &input[3], &input[4], &input[5], &input[6], &input[7], &output);
      for(int i=0; i<tree.getM(); ++i){
          // binned_input[i] = (int)(input[i]-min_val[i])%64;
          binned_input[i] = (int)(input[i])%64;
      }
      // temp = {id, binned_input, output*std::rand()};
      temp = {binned_input, output*std::rand()/(RAND_MAX + 1u)/1000};
      tree.data.push_back(temp);
      id++;
  }

  fclose(train_file);

  // struct featInfo init_hists = {{0.0}, {0.0}, {0.0}};
  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
      init_hists.label_hist[i] = 0.0;
      init_hists.count_hist[i] = 0.0;
  }
  vector<featInfo> hists;
  for(int i=0; i<tree.getM(); ++i) {
    hists.push_back(init_hists);
  }
  vector<int> inst_id;
  for(unsigned i=0; i<tree.data.size(); ++i) {
    inst_id.push_back(i);
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  // struct TNode node = {2.3, 3.5, data, nullptr, nullptr, hists, init_info};
  struct TNode node = {inst_id, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}

