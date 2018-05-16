#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
using namespace std;

struct instance {
  int id;
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
  float hess_hist[64];
  float grad_hist[64];
};

struct TNode {
  float hess;
  float grad;
  std::vector<instance> inst;
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
    struct featInfo init_hists = {{0.0}, {0.0}};
    for(int i=0; i<64; ++i){
      init_hists.grad_hist[i] = 0.0;
      init_hists.hess_hist[i] = 0.0;
    }
    vector<featInfo> hists;
    for(int i=0; i<getM(); ++i) {
      hists.push_back(init_hists);
    }

    vector<instance> data;
    struct SplitInfo init_info = {0, 0, 0.0};
  
    // TNode temp = {0.9, 0.9, data, nullptr, nullptr, hists, init_info};
    TNode temp = {2.1, 3.5, data, nullptr, nullptr, hists, init_info};
    return temp;
  }

  void build_tree(struct TNode* node){
    float entr;
    float max_entr;
    float temp;
    TNode child1[10];
    TNode child2[10];
    
    vector<TNode*> cur_nodes;
    cur_nodes.push_back(node);
    // int total_nodes = 10;
    // int cur_depth = 0;
    // while(cur_depth<10) {
    for(unsigned node_id=0; node_id<cur_nodes.size() && node_id<10; ++node_id){
      max_entr = 0.0;
      temp = 0.0;
      node = cur_nodes[node_id];
      cout << "Came to split node with instances: " << node->inst.size() << " with split feat id: " << node->info.split_feat_id << " and thres: " << node->info.thres << "\n";
      depth++;
      child1[node_id] = init_node();
      child2[node_id] = init_node();
      node->child1 = &child1[node_id];
      node->child2 = &child2[node_id];


      // create_hist(node);
      for(int j=1; j<M; ++j) {
        for(unsigned int i=0; i<((node->inst).size()); ++i) {
           // node.feat_hists[j].hist[node.inst[i].f[j]]++;
           node->feat_hists[j].hess_hist[node->inst[i].f[j]] += node->hess;
           node->feat_hists[j].grad_hist[node->inst[i].f[j]] += node->grad;
        }
      }

      for(int j=0; j<M; ++j) {
        for(unsigned int i=0; i<64; ++i) {
          entr = 0.0;
          for(unsigned int inst_id=0; inst_id<node->inst.size(); ++inst_id) {

            if(node->feat_hists[j].grad_hist[i] && node->feat_hists[j].hess_hist[i]){
              temp = node->feat_hists[j].grad_hist[i]/node->feat_hists[j].hess_hist[i];
            }
            assert(node->inst[inst_id].f[j]<64 && "unexpected feat value\n");
            entr += node->feat_hists[j].hess_hist[i] * pow((node->inst[inst_id].f[j] + temp),2);
            temp = 0.0;
          }
          // entr = entr/(pow(64,2)*node->inst.size());
         if(entr>max_entr){
            max_entr = entr;
            node->info.split_feat_id = j;
            node->info.thres = i;
          } 
        }
      }
      struct instance temp2;
      for(unsigned int i=0; i<node->inst.size(); ++i) {
        temp2 = node->inst[i];
        if(node->inst[i].f[node->info.split_feat_id] <= node->info.thres) {
          // allot to child1
          node->child1->inst.push_back(temp2);
        } else {
          node->child2->inst.push_back(temp2);
        }
      }

      if(node->child1->inst.size() > min_children){
        cur_nodes.push_back(node->child1);
        cout << "size1: " << node->child1->inst.size() <<  "\n";
      }
      if(node->child2->inst.size() > min_children){
        cur_nodes.push_back(node->child2);
        cout << "size2: " << node->child2->inst.size() <<  "\n";
      }
   }
 }
};

int main() {
  DecisionTree tree;
  vector<instance> data; // load directly from your data

  // tree.init_data(65536,8);
  tree.init_data(10000,8);

  // N = 3; M = 2;
  vector<float> input; // binned: I should do binning in CGRA? No, it is required only once
  vector<int> binned_input; // binned: I should do binning in CGRA? No, it is required only once
  for(int i=0; i<tree.getM(); ++i) {
    input.push_back(0.0);
    binned_input.push_back(0.0);
  }
  float output; int id=0;
  instance temp = {1, binned_input, 100};
  FILE* train_file = fopen("train_data_short.csv", "r");
  char lineToRead[5000];
  if(!train_file){
    printf("Error opening file\n");
  }

  float min_val[tree.getM()];
  for(int i=0; i<tree.getM(); ++i){
      min_val[i] = 0;
  }
  // min_val[0] = 0;
  // min_val[1] = 4;
  // min_val[2] = 0;
  // min_val[3] = 6886;
  // min_val[4] = 100; 
  // min_val[5] = 500; 
  // min_val[6] = 1000; 
  // min_val[7] = 0; 
  while(fgets(lineToRead, 500000, train_file) != NULL) {
      if(*lineToRead==',') { continue; }
      sscanf(lineToRead, "%f,%f,%f,%f,%f,%f,%f,%f,%f", &input[0], &input[1], &input[2], &input[3], &input[4], &input[5], &input[6], &input[7], &output);
      for(int i=0; i<tree.getM(); ++i){
          binned_input[i] = (int)(input[i]-min_val[i])%64;
      }
      temp = {id, binned_input, output};
      data.push_back(temp);
      id++;
  }

  fclose(train_file);

  /*
  data.push_back(temp);
  input[0] = 7; input[1] = 25;
  temp = {2, input, 90};
  data.push_back(temp);
  input[0] = 9; input[1] = 35;
  temp = {3, input, 95};
  data.push_back(temp);
  */
/*
  float inhist1[64];
  float inhist2[64];

  for(int i=0; i<64; ++i){
      inhist1[i] = 0.0;
      inhist2[i] = 0.0;
  }
  */
  struct featInfo init_hists = {{0.0}, {0.0}};
  for(int i=0; i<64; ++i){
      init_hists.grad_hist[i] = 0.0;
      init_hists.hess_hist[i] = 0.0;
  }
  // struct featInfo init_hists = {inhist, inhist2};
  vector<featInfo> hists;
  for(int i=0; i<tree.getM(); ++i) {
    hists.push_back(init_hists);
  }

  struct SplitInfo init_info = {0, 0, 0.0};
  struct TNode node = {2.3, 3.5, data, nullptr, nullptr, hists, init_info};
  struct TNode* root = &node;

  tree.build_tree(root);

  return 0;
}

