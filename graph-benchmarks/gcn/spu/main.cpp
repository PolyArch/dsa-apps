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
#include "gcn.dfg.h"
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include "assert.h"
#define VTYPE float

using namespace std;

// #define NODES 19719
#define rate 0.073
#define FEAT_LEN 32 // 64 // 8 // 64
#define output_fm (FEAT_LEN) // 128 // 16 // 3
#define VEC_LEN 16

#define GCN_LAYERS 1 // 6 // 12
#define LADIES_SAMPLE 16 // 64 // 128 // 512 // 256 // 512
#define LADIES_EDGES 4096

// this should be strictly greater than C
#define NUM_THREADS 2 // 8
#define NUM_VERT_PER_THREAD (LADIES_SAMPLE/NUM_THREADS)

using namespace std;

// Barrier variable
pthread_barrier_t barr;

struct edge_info {
  uint32_t dst_id;
  // uuint32_t64_t wgt;
};

#define NUM_BATCH 1

struct gcn_info {
  uint32_t tid;
  uint32_t edge_list[E]; 
  VTYPE wgt[E];
  uint32_t vertex_ptr[V+1];
  // vector<uint32_t> sampled_edge_list[GCN_LAYERS];
  uint32_t sampled_edge_list[GCN_LAYERS][LADIES_EDGES];
  uint32_t sampled_offset[GCN_LAYERS][LADIES_SAMPLE+1];
 
};

void mv(long tid, uint32_t (&edge_list)[E], VTYPE (&wgt)[E], uint32_t (&vertex_ptr)[V+1], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1]) {

  // TODO: use the correct value of feature map
  static uint32_t feature_map[2][LADIES_SAMPLE][FEAT_LEN];
  // TODO: add it while doing matrix multiply
  // VTYPE weights[GCN_LAYERS][FEAT_LEN][output_fm];
  
  uint32_t start_col = tid*NUM_VERT_PER_THREAD;
  uint32_t end_col = (tid+1)*NUM_VERT_PER_THREAD;
  int cur_active_vert=NUM_VERT_PER_THREAD;

  for(int b=0; b<NUM_BATCH; ++b) {

    for(uint32_t l=0; l<GCN_LAYERS; ++l) {

     uint32_t inc_edges = sampled_offset[l][end_col]-sampled_offset[l][start_col];

     // TODO: make sure const works num elems are 0
      int factor = FEAT_LEN/VEC_LEN;
      // SS_2D_CONST(P_gcn_acc_ctrl, 0, factor-1, 1, 1, NUM_VERT_PER_THREAD);
      // SS_2D_CONST(P_gcn_acc_ctrl, 0, factor*(degree-1), 1, factor-1, NUM_VERT_PER_THREAD);
      // SS_CONST(P_gcn_factor, factor, NUM_VERT_PER_THREAD);
      for(unsigned k=start_col; k<end_col; ++k) {
        int degree = sampled_offset[l][k+1]-sampled_offset[l][k];
        if(degree==0) --cur_active_vert;
        if(degree>1) {
        SS_2D_CONST(P_gcn_acc_ctrl, 0, factor*(degree-1), 1, factor, 1);
        } else {
          if(degree>0)
            SS_CONST(P_gcn_acc_ctrl, 1, factor);
        }
      }

      // this is to calculate the degree (offset[start_col+1]-offset[start_col]
      SS_CONST(P_gcn_offset_list0,sampled_offset[l][start_col],1);
      SS_ADD_PORT(P_gcn_offset_list0);
      SS_DMA_READ(&sampled_offset[l][start_col+1], 4, 4, NUM_VERT_PER_THREAD-1, P_gcn_offset_list1);
      SS_CONST(P_gcn_offset_list1,sampled_offset[l][end_col],1);

      // accessing the dst_id (src, src+1, src+2)
      SS_CONFIG_INDIRECT(T32,T32,4,1); // multiplier for offset
      SS_INDIRECT_2D(P_gcn_start_ind, &sampled_edge_list[l][0], NUM_VERT_PER_THREAD, 4, 4, P_gcn_row_size, P_IND_1);

      // similar to indirect_2d but for indirect scratch (fixed length
      // -- this is just to support random datatype)
      SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN);
      SS_INDIRECT_SCR(P_IND_1, 0, inc_edges, P_gcn_feat);

      // FIXME: Oh, this won't be available when degree is 0 (preprocess to
      // remove)
      SS_DMA_WRITE(P_gcn_alpha, 8, 8, cur_active_vert*FEAT_LEN/2, &feature_map[0][0][0]);

      // FIXME: does it wait for all prior insts to be completed or only ss?
      SS_WAIT_ALL();
      SS_GLOBAL_WAIT(NUM_THREADS);
    }



  
  if(b>0) { // TODO: this will require to access the original graph
    static uint32_t sampled_nodes[LADIES_SAMPLE][GCN_LAYERS+1];
    vector<uint32_t> sampled_edge_list_next[GCN_LAYERS];
    static uint32_t sampled_offset_next[GCN_LAYERS][LADIES_SAMPLE+1];
    
    uint32_t i,j,l;
    for(i=0; i<LADIES_SAMPLE; ++i) {
      sampled_nodes[i][GCN_LAYERS] = rand()%V;
    }

    for(l=GCN_LAYERS-1; l>=0; --l) {

      float prob[V]; uint32_t fnorm; float cumsum=0;
      // uint32_t num_non_zero_values=0,
      uint32_t non_zero_index=-1;
      vector<uint32_t> candidate_vid;

      for(i=0; i<V; ++i) prob[i]=0;
      // Q.P (256xV)

      for(i=0; i<LADIES_SAMPLE; ++i) { // reuse on the graph
        uint32_t ind = sampled_nodes[i][l+1];
        for(j=vertex_ptr[ind]; j<vertex_ptr[ind+1]; ++j) {
          prob[edge_list[j]] += pow(wgt[j]*1,2);
        }
      }

      cout << "Done the probab calc\n";

      // frobeneous norm = trace(A*A)
      fnorm = sqrt(E/2); // assuming undirected graph with 1 side edges=E
      for(i=0; i<V; ++i) prob[i]/=(float)fnorm;

         // fill sample nodes using the above probabiity
      // Step1: find cumsum
      // TODO: how to do cumsum effectively using multicore?
      for(i=1; i<V; ++i) {
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
      cout << "Done the candidate calc\n";
      // cout << "Came out of tid\n";
      //  uint32_t rc2 = pthread_barrier_wait(&barr2);
      // cout << "Came out of barrier\n";

        // cout << "Number of candidates: " << non_zero_index << endl;
        float range = prob[non_zero_index]-prob[0];
        // cout << "start: " << prob[0] << " range: " << range << endl;

        // FIXME: error here...can't find
        uint32_t bstart=0, bend=non_zero_index;
        uint32_t a, mid; float x;


        for(i=0; i<LADIES_SAMPLE; ++i) {
        // for(i=start_col; i<end_col; ++i) { // multi-threaded
          a = rand()%100;
          x = (a/(float)100)*range;
          x += prob[0];
          // binary search
          bstart=0; bend=non_zero_index;
          mid = 0;
          uint32_t iter=0;
          while(1) {
            // cout << "prob[mid]: " << prob[mid] << " x: " << x << endl;
            mid = (bstart+bend)/2;
            if(prob[mid]>x && prob[mid-1]<=x) break;
            if(prob[mid]>x) bend = mid-1;
            else bstart = mid+1;
            iter++;
            if(iter>100) cout << "FATAL: CAN'T FIND" << endl;
          }
          sampled_nodes[i][l] = candidate_vid[mid];
        }

        cout << "Done extracting nodes, unique not done?\n";

        // layer-dependent laplacian matrix
        float interm[LADIES_SAMPLE];

        if(1) { // tid==0) { // TODO: can we do better?
          for(i=0; i<LADIES_SAMPLE; ++i) {
            interm[i] = prob[sampled_nodes[i][l]];
          }

          uint32_t e=0, ind;
          for(i=0; i<LADIES_SAMPLE; ++i) { // first row of Q.P
            ind = sampled_nodes[i][l+1];
            sampled_offset_next[l][i] = e;
            for(j=0; j<LADIES_SAMPLE; ++j) {
              uint32_t val=0;
              for(uint32_t k=vertex_ptr[ind]; k<vertex_ptr[ind+1]; ++k) {
                if(edge_list[k]==sampled_nodes[j][l]) {
                  val = interm[j]*wgt[k];
                  sampled_edge_list_next[l].push_back(j); // this doesn't allow parallelization
                  ++e;
                }
              }
            }
          }
          // cout << "Number of edges at layer: " << l << " is: " << e << endl;
          sampled_offset_next[l][LADIES_SAMPLE]=e;
        }
      }
      cout << "norm done\n";
    }
  }

}


void read_adjacency_matrix(uint32_t (&edge_list)[E], VTYPE (&wgt)[E],
                           uint32_t (&vertex_ptr)[V+1]) {

  cout << "start reading graph input file!\n";
  
  // FILE* graph_file = fopen("datasets/citeseer_csr.txt", "r");
  // FILE* graph_file = fopen("datasets/flickr_csr", "r");
 // FILE* graph_file = fopen("/home/vidushi.dadu/graphsim/datasets/directed_power_law/flickr_csr", "r");
 string str(csr_file);

 FILE* graph_file = fopen(str.c_str(), "r");


 // FILE* graph_file = fopen("datasets/flickr_csr", "r");
  // FILE* graph_file = fopen("datasets/pubmed_csr.txt", "r");

  char linetoread[5000];

  vertex_ptr[0]=0;
  uint32_t prev_offset=0;
  uint32_t e=-1, prev_v=0; // indices start from 1
  bool ignore=false;
  while(fgets(linetoread, 5000, graph_file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    uint32_t src, dst, x;
    // iss >> src >> dst >> x;
    iss >> src >> dst;
    src = src-1;
    dst = dst-1;

    // cout << "src: " << src << " dst: " << dst << " x: " << x << endl;

    // uint32_t degree = e-vertex_ptr[prev_v];

    if(!ignore) {
      edge_list[++e]=dst;
      wgt[e]=x;
      // cout << "Current edge: " << e << " with dst: " << dst << " and src: " << src << " prev_v: " << prev_v << endl;
    }

    if(src!=prev_v) {
      // assert(degree<=SAMPLE_SIZE);
      // cout << "Current node degree: " << degree << endl;
      vertex_ptr[prev_v+1]=e;
      // cout << (prev_v+1) << " OFFSET: " << e << endl;
      uint32_t k=prev_v+1;
      while(vertex_ptr[--k]==0 && k>0) {
        vertex_ptr[k]=prev_offset;
      }
      prev_offset=e;
      prev_v=src;
      // cout << "index: " << (src) << " value: " << e << endl;
    }
    // cout << _neighbor[e].wgt << " " << _neighbor[e].dst_id << " " << _offset[prev_v-1] << endl;
  }
  vertex_ptr[V] = E;
  uint32_t k=V;
  while(vertex_ptr[--k]==0 && k>0) { // offset[0] should be 0
    vertex_ptr[k]=prev_offset;
  }
  fclose(graph_file);
  cout << "Done reading graph file!\n";
  // FIXME
  /*for(uint32_t i=0; i<=V; ++i) {
      cout << "vertex_ptr: " << vertex_ptr[i] << endl;
  }*/
}

// considering it to be dense (classes~100?)
void fill_input_fm(uint32_t (&feature_map)[V][FEAT_LEN][2]) {
  for(uint32_t i=0; i<V; ++i) {
    for(uint32_t j=0; j<FEAT_LEN; ++j) {
      feature_map[i][j][0] = 1;
    }
  }
}

// considering this also to be dense (unpruned neural network)
void fill_weights(VTYPE (&weights)[GCN_LAYERS][FEAT_LEN][output_fm]) {
  for(uint32_t k=0; k<GCN_LAYERS; ++k) {
    for(uint32_t i=0; i<FEAT_LEN; ++i) {
      for(uint32_t j=0; j<output_fm; ++j) {
        weights[k][i][j] = 1;
      }
    }
  }
}

void *entry_point(void *info) {

  long tid = ((struct gcn_info*)info)->tid;
  
  SS_CONFIG(gcn_config, gcn_size);
  SS_GLOBAL_WAIT(NUM_THREADS);

  // Synchronization pouint32_t
  uint32_t rc = pthread_barrier_wait(&barr);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
  {
    printf("Could not wait on barrier\n");
  }

  begin_roi();
  mv(tid, ((struct gcn_info*)info)->edge_list, ((struct gcn_info*)info)->wgt, ((struct gcn_info*)info)->vertex_ptr, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
 
  end_roi();
  sb_stats();

  return NULL;
}

void perform_sampling(uint32_t (&edge_list)[E], VTYPE (&wgt)[E],
        uint32_t (&vertex_ptr)[V+1], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES]) {
  cout << "Started doing ladies sampling for first batch\n";

  uint32_t i,j;
  static uint32_t sampled_nodes[LADIES_SAMPLE][GCN_LAYERS+1];

  for(uint32_t i=0; i<LADIES_SAMPLE; ++i) {
    sampled_nodes[i][GCN_LAYERS] = rand()%V;
  }

  for(int l=GCN_LAYERS-1; l>=0; --l) {

    float prob[V];

    for(i=0; i<V; ++i) prob[i]=0;
    // Q.P (256xV)

    for(i=0; i<LADIES_SAMPLE; ++i) { // reuse on the graph
      uint32_t ind = sampled_nodes[i][l+1];
      for(j=vertex_ptr[ind]; j<vertex_ptr[ind+1]; ++j) {
        prob[edge_list[j]] += pow(wgt[j]*1,2);
      }
    }

     // frobeneous norm = trace(A*A)
     uint32_t fnorm = sqrt(E/2); // assuming undirected graph with 1 side edges=E
     for(i=0; i<V; ++i) prob[i]/=(float)fnorm;

     cout << "prob calc\n";

     // uint32_t num_non_zero_values=0,
     uint32_t non_zero_index=-1;
     vector<uint32_t> candidate_vid;
     float cumsum=0;

      // fill sample nodes using the above probabiity
      // Step1: find cumsum
      // TODO: how to do cumsum effectively using multicore?
      for(i=1; i<V; ++i) {
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

      cout << "Number of candidates: " << non_zero_index << endl;
      float range = prob[non_zero_index]-prob[0];
      // cout << "start: " << prob[0] << " range: " << range << endl;

      uint32_t bstart=0, bend=non_zero_index;
      uint32_t a, mid; float x;

      for(i=0; i<LADIES_SAMPLE; ++i) {
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

      cout << "Nodes sampled\n";

      // layer-dependent laplacian matrix
      float interm[LADIES_SAMPLE];

      for(i=0; i<LADIES_SAMPLE; ++i) {
        interm[i] = prob[sampled_nodes[i][l]];
      }

      uint32_t e=0, ind;
      for(i=0; i<LADIES_SAMPLE; ++i) { // first row of Q.P
        ind = sampled_nodes[i][l+1];
        sampled_offset[l][i] = e;
        for(j=0; j<LADIES_SAMPLE; ++j) {
          uint32_t val=0;
          for(uint32_t k=vertex_ptr[ind]; k<vertex_ptr[ind+1]; ++k) {
            if(edge_list[k]==sampled_nodes[j][l]) {
              val = interm[j]*wgt[k];
              // sampled_edge_list[l].push_back(j); // this doesn't allow parallelization
              sampled_edge_list[l][e]=j; // this doesn't allow parallelization
              ++e;
            }
          }
        }
      }
      // cout << "Number of edges at layer: " << l << " is: " << e << endl;
      sampled_offset[l][LADIES_SAMPLE]=e;
    }
    cout << "Done doing ladies sampling for first batch\n";
    for(int k=0; k<=LADIES_SAMPLE; ++k) {
      cout << "offset at k: " << sampled_offset[0][k] << endl; 
    }
}

int main() {

  struct gcn_info *info = (struct gcn_info*)malloc(sizeof(struct gcn_info));


  read_adjacency_matrix(info->edge_list, info->wgt, info->vertex_ptr);

  // TODO: add later
  // static VTYPE weights[GCN_LAYERS][FEAT_LEN][output_fm];
  // static uint32_t feature_map[V][FEAT_LEN][2];

  // fill_input_fm(feature_map);
  // fill_weights(weights);

  // Get laplacian matrix P = D-1/2AD-1/2
  // d is degree of all nodes
  // 1st multiplication is 1/sqrt(outgoing degree) * (if incoming edge to it) -- VxV (oh just scaling by the corresponding factor)
  for(uint32_t i=0; i<V; ++i) { // all vertices
    uint32_t out_degree = info->vertex_ptr[i+1]-info->vertex_ptr[i];
    for(uint32_t j=info->vertex_ptr[i]; j<info->vertex_ptr[i+1]; ++j) { // index of the corresponding edge
      uint32_t dst_id = info->edge_list[j];
      uint32_t out_degree_of_dest = info->vertex_ptr[dst_id+1] - info->vertex_ptr[dst_id];
      out_degree += 1;
      out_degree_of_dest += 1;
      info->wgt[j]=1*1/(float)(sqrt(out_degree)*sqrt(out_degree_of_dest));
    }
  }

  int scratch_space = LADIES_SAMPLE*FEAT_LEN*4/NUM_THREADS;
  assert(scratch_space<16384 && "required scratch space is more than available, increase cores");
  assert(FEAT_LEN%VEC_LEN==0 && "feat_len should be a multiple of vec_len");
  assert(FEAT_LEN%16==0 && "currently only support 64-byte multiple wide type");

  perform_sampling(info->edge_list, info->wgt, info->vertex_ptr, info->sampled_offset, info->sampled_edge_list);
  cout << "Sampling done\n";
  /*for(int k=0; k<LADIES_SAMPLE; ++k) {
    cout << "offset at k: " << k << " is: " << info->sampled_offset[0][k] << endl;
  }
  for(int k=0; k<info->sampled_offset[0][LADIES_SAMPLE]; ++k) {
    cout << "dst id at k: " << k << " is: " << info->sampled_edge_list[0][k] << endl;
  }*/
  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  uint32_t rc;
  long t;
  for(t=0;t<NUM_THREADS;t++){
    printf("In main: creating thread %ld\n", t);
    info->tid = t;
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)info);
    // rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      return 0;
    }
  }

  for(uint32_t i = 0; i < NUM_THREADS; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return 0;
    }
  }
  return 0;
}

