#include <iostream>
#include <vector>
#include <math.h>
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
    // int id;
    // int n; // number of data
    float hess;
    float grad;
    std::vector<instance> inst;
    // int child[2];
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

    void init_data() {
        min_children = 2;
        max_depth = 10;
        // take this as input
        N = 0;
        M = 0;

    }


    void create_hist(struct TNode* node) {
        for(int j=0; j<M; ++j) {
            for(unsigned int i=0; i<((node->inst).size()); ++i) {
               // node.feat_hists[j].hist[node.inst[i].f[j]]++;
               node->feat_hists[j].hess_hist[node->inst[i].f[j]] += node->hess;
               node->feat_hists[j].grad_hist[node->inst[i].f[j]] += node->grad;
            }
        }
    }

    // find <feat_id, thres> combination
    void find_split(struct TNode* node) {
        float entr = 0;
        float min_entr = 1;

        create_hist(node);

        // check over all features: histograms corresponding to all features
        for(int j=0; j<M; ++j) {
            for(unsigned int i=0; i<64; ++i) {
                // sum over all instances
                entr = 0;
                for(unsigned int inst_id=0; inst_id<node->inst.size(); ++inst_id) {
                    entr += node->feat_hists[i].hess_hist[i] * pow((node->inst[inst_id].f[j] + node->feat_hists[j].grad_hist[i]/node->feat_hists[j].hess_hist[i]),2);
                }
                if(entr<min_entr){
                    min_entr = entr;
                    node->info.split_feat_id = j;
                    node->info.thres = i;
                }    
            }
        }
    }

    void split(struct TNode* node) {
        // need to initialize other information here: have a nice function for this!
        // struct TNode* child1 = (TNode*)malloc(sizeof(TNode*));
        // struct TNode* child2 = (TNode*)malloc(sizeof(TNode*));

        // check over all instances for their particular feat_id and allot to
        // child nodes
        for(unsigned int i=0; i<node->inst.size(); ++i) {
            struct instance temp = node->inst[i]; // Should it be a pointer
            if(node->inst[i].f[node->info.split_feat_id] <= node->info.thres) {
                // allot to child1
                node->child1->inst.push_back(temp);
            }
            else {
                node->child2->inst.push_back(temp);
            }

        }

        if(node->child1->inst.size() > min_children)
            build_tree(node->child1);
        if(node->child2->inst.size() > min_children)
            build_tree(node->child2);


    }

    void build_tree(struct TNode* node){
        node->child1 = (TNode*)malloc(node->inst.size()*sizeof(TNode*)); // Allocating maximum size
        node->child2 = (TNode*)malloc(node->inst.size()*sizeof(TNode*)); // Allocating maximum size
        find_split(node);
        split(node);
    }

    int main() {
        vector<instance> data; // load directly from your data

        init_data();

        N = 3; M = 2;
        vector<int> input; // binned: I should do binning in CGRA? No, it is required only once
        input.push_back(3);
        input.push_back(16);
        instance temp = {1, input, 100};
        data.push_back(temp);
        input[0] = 7; input[1] = 25;
        temp = {2, input, 90};
        data.push_back(temp);
        input[0] = 9; input[1] = 35;
        temp = {3, input, 95};
        data.push_back(temp);


        
        struct featInfo init_hists = {{}, {}};
        vector<featInfo> hists;
        for(int i=0; i<M; ++i) {
            hists.push_back(init_hists);
        }

        struct SplitInfo init_info = {0, 0, 0.0};
        struct TNode node = {0.0, 0.0, data, nullptr, nullptr, hists, init_info};
        struct TNode* root = &node;

        build_tree(root);

        return 0;
    }
    

};
