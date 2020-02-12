#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stack>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <unordered_map>
#include <bitset>
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include <immintrin.h>
#include "assert.h"
// #include "mkl.h"
#define VTYPE float

#define LADIES_GCN 1

using namespace std;

// citeseer
/*
#define NODES 3327
#define EDGES 4732
#define classes 6
#define features 16 // 3703
#define rate 0.073
#define output_fm 6
*/

// pubmed

// flickr
#define NODES  820878 //3327  //
#define EDGES  9837214 // 4731 // 19674428 //  // 19674428 // 9837214

// pubmed
// #define NODES  19719 //3327  //
// #define EDGES  88673 // 4731 // 19674428 //  // 19674428 // 9837214

// citeseer
// #define NODES  3327 //3327  //
// #define EDGES 93328 // 4731 // 19674428 //  // 19674428 // 9837214

// cit-Patents
// #define NODES  3774768
// #define EDGES 16518948




#define classes 1
#define features 256 // 500 // 16 // 500
#define rate 0.073
#define output_fm 256 // 128 // 16 // 3

vector<int> sampled_edge_list[GCN_LAYERS]; // ine new_edge_list[...] is cudaMemCpy not allowed?

// #define NODES 65536
#define SAMPLE_SIZE 25
#define SAGE_EDGES (NODES*SAMPLE_SIZE)

vector<int> adj_list[NODES];
// need src->dst to dst->src 
void convert_push_to_pull(int (&edge_list)[EDGES], VTYPE (&wgt)[EDGES],
                           int (&vertex_ptr)[NODES+1]) {

  vertex_ptr[0]=0;
  for(int i=0; i<NODES; ++i) {
    vertex_ptr[i+1]=adj_list[i].size();
    for(unsigned j=0; j<adj_list[i].size(); ++j) {
      int edge_id = j+vertex_ptr[i];
      assert(edge_id<EDGES);
      edge_list[edge_id] = adj_list[i][j];
    }
  }
  for(int i=0; i<NODES; ++i) adj_list[i].clear();
}

void read_adjacency_matrix(int (&edge_list)[EDGES], VTYPE (&wgt)[EDGES],
                           int (&vertex_ptr)[NODES+1]) {

  cout << "start reading graph input file!\n";
  
 FILE* graph_file = fopen("../../pagerank/datasets/flickr_csr", "r");
 // FILE* graph_file = fopen("datasets/pubmed_und_csr", "r");
 // FILE* graph_file = fopen("datasets/citeseer_und_csr.txt", "r");
 // FILE* graph_file = fopen("/home/vidushi/graphsim/datasets/undirected/cit-Patents.mtx", "r");

  char linetoread[5000];

  vertex_ptr[0]=0;
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    int src, dst, x;
    iss >> src >> dst >> x;
    adj_list[src].push_back(dst);
    // adj_list[dst].push_back(src);
  }
  fclose(graph_file);
  cout << "Done reading graph file!\n";
  // FIXME
  /*for(int i=0; i<=NODES; ++i) {
      cout << "vertex_ptr: " << vertex_ptr[i] << endl;
  }*/
}

// considering it to be dense (classes~100?)
void fill_input_fm(int (&feature_map)[NODES][features][2]) {
  for(int i=0; i<NODES; ++i) {
    for(int j=0; j<features; ++j) {
      feature_map[i][j][0] = 1;
    }
  }
}

// considering this also to be dense (unpruned neural network)
void fill_weights(VTYPE (&weights)[GCN_LAYERS][features][output_fm]) {
  for(int k=0; k<GCN_LAYERS; ++k) {
    for(int i=0; i<features; ++i) {
      for(int j=0; j<output_fm; ++j) {
        weights[k][i][j] = 1;
      }
    }
  }
}


int sampled_nodes[LADIES_SAMPLE][GCN_LAYERS+1];

void perform_sampling(int (&edge_list)[EDGES], VTYPE (&wgt)[EDGES],
        int (&sampled_vertex_ptr)[GCN_LAYERS][LADIES_SAMPLE+1],
        // vector<int> (&sampled_edge_list)[GCN_LAYERS],
        int (&vertex_ptr)[NODES+1]) {

  int i,j;

  // random selected nodes are higher probability to low degree?
  for(i=0; i<LADIES_SAMPLE; ++i) {
    srand(i);
    sampled_nodes[i][GCN_LAYERS] = rand()%NODES;
  }

  /*for(int k=0; k<LADIES_SAMPLE; ++k) {
    cout << "Node selected: " << sampled_nodes[k][GCN_LAYERS] << endl;
  }*/

  vector<int> sampled_adj_list[GCN_LAYERS][LADIES_SAMPLE];
  for(int l=GCN_LAYERS-1; l>=0; --l) {

    float prob[NODES] = {0};

    for(i=0; i<NODES; ++i) prob[i]=0;
    // Q.P (256xV)

    for(i=0; i<LADIES_SAMPLE; ++i) { // reuse on the graph
      int ind = sampled_nodes[i][l+1];
      for(j=vertex_ptr[ind]; j<vertex_ptr[ind+1]; ++j) {
        prob[edge_list[j]] += 1; // pow(wgt[j]*1,2);
      }
    }

    /*for(i=0; i<NODES; ++i) {
      if(prob[i]!=0)
      cout << "i: " << i << " prob: " << prob[i] << endl;
    }*/

     // frobeneous norm = trace(A*A)
     // int fnorm = sqrt(EDGES/2); // assuming undirected graph with 1 side edges=E
     // for(i=0; i<NODES; ++i) prob[i]/=(float)fnorm;

     // int num_non_zero_values=0,
     int non_zero_index=-1;
     vector<int> candidate_vid;
     float cumsum=0;

      // fill sample nodes using the above probabiity
      // Step1: find cumsum
      // TODO: how to do cumsum effectively using multicore?
      for(i=1; i<NODES; ++i) {
        // cout << "prob at i: " << prob[i] << endl;
        cumsum = prob[i] + prob[i-1];
        if(prob[i]!=0) {
          prob[++non_zero_index] = cumsum;
          // cout << "Allocated prob: " << prob[non_zero_index] << endl;
          candidate_vid.push_back(i);
        }
        prob[i]=cumsum;
        // cout << "Cumulative probability for node i: " << i << " " << prob[i] << endl;
      }

      // cout << "Number of candidates: " << non_zero_index << endl;
      float range = prob[non_zero_index]-prob[0];
      // cout << "start: " << prob[0] << " range: " << range << endl;

      // FIXME: error here...can't find
      int bstart=0, bend=non_zero_index;
      int a, mid; float x;

      for(i=0; i<LADIES_SAMPLE; ++i) {
        srand(i);
        a = rand()%100;
        x = (a/(float)100)*range;
        x += prob[0];
        // binary search
        bstart=0; bend=non_zero_index;
        mid = 0;
        while(1) {
          // cout << "prob[mid]: " << prob[mid] << " x: " << x << endl;
          mid = (bstart+bend)/2;
          if(prob[mid]>x && prob[mid-1]<=x) break;
          if(prob[mid]>x) bend = mid-1;
          else bstart = mid+1;
        }
        sampled_nodes[i][l] = candidate_vid[mid];
      }

      /*for(int k=0; k<LADIES_SAMPLE; ++k) {
        cout << "Node selected2: " << sampled_nodes[k][l] << endl;
      }*/
      // layer-dependent laplacian matrix
      float interm[LADIES_SAMPLE];

      for(i=0; i<LADIES_SAMPLE; ++i) {
        interm[i] = prob[sampled_nodes[i][l]];
      }

      int e=0, ind;
      for(i=0; i<LADIES_SAMPLE; ++i) { // first row of Q.P
        ind = sampled_nodes[i][l+1];
        sampled_vertex_ptr[l][i] = e;
        for(j=0; j<LADIES_SAMPLE; ++j) {
          int val=0;
          for(int k=vertex_ptr[ind]; k<vertex_ptr[ind+1]; ++k) {
            if(edge_list[k]==sampled_nodes[j][l]) {
              val = interm[j]*wgt[k];
              sampled_adj_list[l][j].push_back(i);
              sampled_edge_list[l].push_back(j); // this doesn't allow parallelization
              ++e;
            }
          }
        }
      }
      cout << "Number of edges at layer: " << l << " is: " << e << endl;
      sampled_vertex_ptr[l][LADIES_SAMPLE]=e;
    }


  // Store the result in output file
  // SAMPLED NODES
  ofstream node_map("sampled_nodes.txt");
  if(node_map.is_open()) {
    for(int l=0; l<GCN_LAYERS; ++l) {
      for(j=0; j<LADIES_SAMPLE; ++j) {
        node_map << sampled_nodes[j][l] << " ";
      }
      node_map << "\n";
    }
  }
  node_map.close();
  // TODO: for pull implementation, this should store the pull version of the
  // graph
  // ADJ MATRIX
  for(int l=0; l<GCN_LAYERS; ++l) {
    sampled_vertex_ptr[l][0]=0;
    for(i=0; i<LADIES_SAMPLE; ++i) {
      sampled_vertex_ptr[l][i+1] = sampled_adj_list[l][i].size() + sampled_vertex_ptr[l][i];
      // cout << "sampled ptr: " << sampled_vertex_ptr[l][i+1] << endl;
      for(unsigned k=0; k<sampled_adj_list[l][i].size(); ++k) {
        int edge_ind = k + sampled_vertex_ptr[l][i];
        sampled_edge_list[l][edge_ind] = sampled_adj_list[l][i][k];
      }
    }
  }
  ofstream adj_new("adj_mat.txt");
  if(adj_new.is_open()) {
    for(int l=0; l<GCN_LAYERS; ++l) {
      // all vertex ptr
      for(i=0; i<=LADIES_SAMPLE; ++i) {
        adj_new << sampled_vertex_ptr[l][i] << " ";
      }
      adj_new << "\n";
      // all destinations in this graph
      for(j=0; j<sampled_vertex_ptr[l][LADIES_SAMPLE]; ++j) {
        adj_new << sampled_edge_list[l][j] << " ";
      }
      adj_new << "\n";
    }
  }
  adj_new.close();
  /*
  int in_degree[LADIES_SAMPLE];
  for(int l=0; l<GCN_LAYERS; ++l) {
    for(i=0; i<LADIES_SAMPLE; ++i) in_degree[i]=0;
    for(i=0; i<=LADIES_SAMPLE; ++i) {
      for(j=sampled_vertex_ptr[l][i]; j<sampled_vertex_ptr[l][i+1]; ++j) {
        // cout << "dest at i: " << i << " " << sampled_edge_list[l][j] << endl;
        in_degree[sampled_edge_list[l][j]]++;
      }
    }
    for(int k=0; k<LADIES_SAMPLE; ++k)
      cout << "node k: " << k  << " " << in_degree[k] << endl;
    }

  for(int l=0; l<GCN_LAYERS; ++l) {
    for(i=0; i<LADIES_SAMPLE; ++i) {
      cout << "out degree node k: " << i  << " " << (sampled_vertex_ptr[l][i+1]-sampled_vertex_ptr[l][i]) << endl;
    }
  }
  */


  for(int l=0; l<GCN_LAYERS; ++l) {
    for(i=0; i<LADIES_SAMPLE; ++i) sampled_adj_list[l][i].clear();
  }
}

int main() {
  static int edge_list[EDGES];
  static VTYPE wgt[EDGES];
  static int vertex_ptr[NODES+1];

  static int sampled_vertex_ptr[GCN_LAYERS][LADIES_SAMPLE+1];


  static VTYPE weights[GCN_LAYERS][features][output_fm];
  static int feature_map[NODES][features][2]; // need to maintain 2 at a time

  read_adjacency_matrix(edge_list, wgt, vertex_ptr);
  convert_push_to_pull(edge_list, wgt, vertex_ptr);
  fill_input_fm(feature_map);
  fill_weights(weights);

  // Get laplacian matrix P = D-1/2AD-1/2
  // d is degree of all nodes
  // 1st multiplication is 1/sqrt(outgoing degree) * (if incoming edge to it) -- VxV (oh just scaling by the corresponding factor)
  for(int i=0; i<NODES; ++i) { // all vertices
    int out_degree = vertex_ptr[i+1]-vertex_ptr[i];
    for(int j=vertex_ptr[i]; j<vertex_ptr[i+1]; ++j) { // index of the corresponding edge
      int dst_id = edge_list[j];
      int out_degree_of_dest =vertex_ptr[dst_id+1] - vertex_ptr[dst_id];
      out_degree += 1;
      out_degree_of_dest += 1;
      wgt[j]=1*1/(float)(sqrt(out_degree)*sqrt(out_degree_of_dest));
    }
  }

  // perform_sampling(edge_list, wgt, sampled_vertex_ptr, vertex_ptr);
  // return 0;

#if LADIES_GCN==1
  for(int t=0; t<1; ++t) {
    begin_roi();
    // perform_sampling(edge_list, wgt, sampled_vertex_ptr, sampled_edge_list, vertex_ptr);
    perform_sampling(edge_list, wgt, sampled_vertex_ptr, vertex_ptr);
    end_roi();
  }
   cout << "Sampling done\n";
   // exit(0);
#endif

  return 0;
}

