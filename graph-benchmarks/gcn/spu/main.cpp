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
#define FEAT_LEN 64
#define VEC_LEN 16

#define GCN_LAYERS 1 // 6 // 12
#define LADIES_SAMPLE 32 // 36 // 32 // 16 // 256 // 16 // 64 // 128 // 512 // 256 // 512
#define LADIES_EDGES 4096

// this should be strictly less than C
#define NUM_THREADS 4 // 8 // 4 // 2 // 8
#define NUM_VERT_PER_THREAD (LADIES_SAMPLE/NUM_THREADS)
#define SCRATCH_SIZE 16384
#define FEAT_PART_SIZE (NUM_VERT_PER_THREAD*FEAT_LEN*4*2) // (SCRATCH_SIZE/2)
#define SCRATCH_BITS 14 // (log2(SCRATCH_SIZE))
#define WGT_CORE 0 // (NUM_THREADS/2)

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


const int factor = FEAT_LEN/VEC_LEN;

// TODO: reuse weights matrix in a better way..
void mm(int l, int cur_active_vert) {
  static uint32_t feature_map[2][LADIES_SAMPLE][FEAT_LEN];
  VTYPE weights[GCN_LAYERS][FEAT_LEN][FEAT_LEN];
  SS_DMA_WRITE(P_gcn_A, 4, 4, cur_active_vert*FEAT_LEN*factor*2, &feature_map[0][0][0]);
  for(int k=0; k<cur_active_vert; ++k) {
    SS_DMA_READ(&weights[l][0][0], 4, 4, FEAT_LEN*FEAT_LEN, P_gcn_weights);
    // SS_SCR_PORT_STREAM(0, 4, 4, FEAT_LEN*FEAT_LEN, P_gcn_weights);
    for(int l=0; l<FEAT_LEN; ++l) {
      SS_SCR_PORT_STREAM(0, 4, 4, FEAT_LEN, P_gcn_agg_feat);
    }
  }
  SS_WAIT_ALL();
}


uint32_t sampled_nodes[GCN_LAYERS+1][LADIES_SAMPLE];
int wgt_rem_loc = WGT_CORE<<SCRATCH_BITS;
uint64_t feat_active_core_mask=0;
uint64_t wgt_active_core_mask=0;

void mv(long tid, uint32_t (&edge_list)[E], VTYPE (&wgt)[E], uint32_t (&vertex_ptr)[V+1], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1]) {

  // TODO: use the correct value of feature map
  // TODO: should be 1 after correct reduction
  static uint32_t feature_map[2][LADIES_SAMPLE][FEAT_LEN][factor];
  // should be read from scratch..
  uint32_t weights[GCN_LAYERS][FEAT_LEN][FEAT_LEN];
  
  uint32_t start_col = tid*NUM_VERT_PER_THREAD;
  uint32_t end_col = (tid+1)*NUM_VERT_PER_THREAD;
  int cur_active_vert=NUM_VERT_PER_THREAD;

  // TODO: this should not be required, wgt matrix won't be in sparse format
  for(unsigned k=start_col; k<end_col; ++k) {
    int degree = sampled_offset[0][k+1]-sampled_offset[0][k];
    if(degree==0) --cur_active_vert;
  }

  begin_roi();
  for(int b=0; b<NUM_BATCH; ++b) {

    for(uint32_t l=0; l<GCN_LAYERS; ++l) {

#if PRELOAD==1
#if POSTLOAD==1
      // SS_ADD_PORT(P_IND_4); // this just stops at wait_all -- as all ports
      // are not empty
#endif
      // TODO: do in the cores participating in the aggregation phase
      SS_DMA_READ(&sampled_nodes[l][start_col], 4, 4, NUM_VERT_PER_THREAD, P_IND_2);
      SS_DCONST(P_IND_3, FEAT_LEN, NUM_VERT_PER_THREAD, T32);
      SS_CONFIG_INDIRECT(T32, T32, 4, 1); // push a value of 4 bytes to this port
      SS_INDIRECT_2D(P_IND_2, &feature_map[0][0][0][0], NUM_VERT_PER_THREAD, 4, 4, P_IND_3, MEM_SCR_PORT);
      SS_SCR_WRITE(MEM_SCR_PORT, NUM_VERT_PER_THREAD*FEAT_LEN*4, 0);

      // TODO: get this to work?

      // SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN);
      // SS_INDIRECT(P_IND_2, &feature_map[0][0][0][0], NUM_VERT_PER_THREAD, MEM_SCR_PORT);

      // TODO: do it in the cores only matrix multiplication (TODO: variable
      // scratch size?)
      /*if(tid==WGT_CORE) { // specifying local location here
        // this core should load its part
        SS_DMA_SCRATCH_LOAD(&weights[l][0][0], 4, 4, FEAT_LEN*FEAT_LEN, 0);
      }*/
      // SS_WAIT_GLOBAL_SCR_WR(); // TODO: should wait on all cores scr write
      SS_WAIT_ALL();
      SS_GLOBAL_WAIT(NUM_THREADS);
#endif

#if AGG==1
      uint32_t inc_edges = sampled_offset[l][end_col]-sampled_offset[l][start_col];
      SS_CONST(P_gcn_factor, factor, NUM_VERT_PER_THREAD);
      SS_2D_DCONST(P_gcn_acc_ctrl, 0, P_gcn_first, 1, P_gcn_last, NUM_VERT_PER_THREAD);

      // this is to calculate the degree (offset[start_col+1]-offset[start_col]
      // No reuse, hence can be accessed from memory (TODO: assuming duplicate
      // sampled graphs, instead of the flag)
      SS_CONST(P_gcn_offset_list0,sampled_offset[l][start_col],1);
      SS_ADD_PORT(P_gcn_offset_list0);
      SS_DMA_READ(&sampled_offset[l][start_col+1], 4, 4, NUM_VERT_PER_THREAD-1, P_gcn_offset_list1);
      SS_CONST(P_gcn_offset_list1,sampled_offset[l][end_col],1);

      // accessing the dst_id (src, src+1, src+2)
      SS_CONFIG_INDIRECT(T32,T32,4,1); // multiplier for offset
      SS_INDIRECT_2D(P_gcn_start_ind, &sampled_edge_list[l][0], NUM_VERT_PER_THREAD, 4, 4, P_gcn_row_size, P_IND_1);

      // TODO: Why did 0 bitvector work? 
      SS_CONFIG_MEM_MAP(FEAT_PART_SIZE,feat_active_core_mask,0);
      SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN);
      SS_INDIRECT_SCR(P_IND_1, 0, inc_edges, P_gcn_feat);

#if MULT==0
      SS_DMA_WRITE(P_gcn_alpha, 8, 8, cur_active_vert*FEAT_LEN/2, &feature_map[0][0][0]);
#endif
#endif
#if SYNC==1
      SS_SCR_WRITE(P_gcn_alpha, FEAT_LEN*4*cur_active_vert, 0);
#else
      // matrix-vector multiplication (active_vert*feat_len)
#if MULT==1
      // scatter data to memory, it should produce the correct number of values at A

      SS_2D_CONST(P_gcn_const, 1, FEAT_LEN-1, 0, 1, cur_active_vert);
      for(int k=0; k<cur_active_vert; ++k) {

        // 65536 + 4*128*128 -- it spans 4 cores
        SS_CONFIG_MEM_MAP(SCRATCH_SIZE,wgt_active_core_mask,0);
        SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN);
        SS_INDIRECT_SCR(P_gcn_running_sum, wgt_rem_loc, FEAT_LEN, P_gcn_weights);

        // TODO: reserved location in the current core (for vertex data, it
        // should use only half of the location)
        // TODO: try buffet here? this is when we require fine-grained barrier, any better way?
        // both recurrence and scratch write?
#if AGG==1
        // TODO: probably want to initiate in other core (for now, can
        // statically do this) -- doesn't need task-based support
        // also fine-grained barrier? (TODO: can i use buffet here?)
        SS_SCR_WRITE(P_gcn_alpha, FEAT_LEN*4, 0);
        SS_WAIT_SCR_WR();
#endif
        SS_SCR_PORT_STREAM(0, 0, 4*FEAT_LEN, FEAT_LEN, P_gcn_agg_feat);
      }

#if POSTLOAD==1 
      // scatter to memory
      SS_DMA_READ(&sampled_nodes[l][start_col], 4, 4, cur_active_vert, P_IND_4);
      SS_DCONST(P_IND_3, FEAT_LEN*2*factor, cur_active_vert, T32);
      SS_CONFIG_INDIRECT(T32, T32, 4, 1);
      SS_INDIRECT_2D_WR(P_IND_4, &feature_map[0][start_col][0][0], cur_active_vert, 4, 4, P_IND_3, P_gcn_A);
 
      // SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN*factor*2);
      // SS_INDIRECT_WR(P_IND_4, &feature_map[0][start_col], cur_active_vert, P_gcn_A);

#else
      SS_DMA_WRITE(P_gcn_A, 4, 4, cur_active_vert*FEAT_LEN*factor*2, &feature_map[0][start_col][0][0]);
#endif


#endif

#endif
      // same number of reads, more writes in outer. More calc. due to no redn
      // SS_CONFIG_ATOMIC_SCR_OP(T32, T32, T32, FEAT_LEN);
      // SS_ATOMIC_SCR_OP(P_gcn_col_id, P_gcn_mult, 0, agg, 0);

      // FIXME: does it wait for all prior insts to be completed or only ss?
      SS_WAIT_ALL();
      // TODO: add print of port-mismatch for this mask as well..
      SS_GLOBAL_WAIT(NUM_THREADS);
#if SYNC==1
      mm(l, cur_active_vert);
      SS_GLOBAL_WAIT(NUM_THREADS);
#endif
    }

  
  if(0) { // b>0) { // TODO: this will require to access the original graph
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
  end_roi();
  sb_stats();
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
void fill_weights(VTYPE (&weights)[GCN_LAYERS][FEAT_LEN][FEAT_LEN]) {
  for(uint32_t k=0; k<GCN_LAYERS; ++k) {
    for(uint32_t i=0; i<FEAT_LEN; ++i) {
      for(uint32_t j=0; j<FEAT_LEN; ++j) {
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

  // begin_roi();
  mv(tid, ((struct gcn_info*)info)->edge_list, ((struct gcn_info*)info)->wgt, ((struct gcn_info*)info)->vertex_ptr, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
 
  // end_roi();
  // sb_stats();

  return NULL;
}


// FIXME: doubt: Are the sampled nodes in each layer unique from the
// other layers? (they are unique at least for the current layer)
void perform_sampling(uint32_t (&edge_list)[E], VTYPE (&wgt)[E],
        uint32_t (&vertex_ptr)[V+1], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES]) {
  cout << "Started doing ladies sampling for first batch\n";

  uint32_t i,j;
  // static uint32_t sampled_nodes[LADIES_SAMPLE][GCN_LAYERS+1];

  for(uint32_t i=0; i<LADIES_SAMPLE; ++i) {
    sampled_nodes[GCN_LAYERS][i] = rand()%V;
  }

  for(int l=GCN_LAYERS-1; l>=0; --l) {

    float prob[V];

    for(i=0; i<V; ++i) prob[i]=0;
    // Q.P (256xV)

    for(i=0; i<LADIES_SAMPLE; ++i) { // reuse on the graph
      uint32_t ind = sampled_nodes[l+1][i];
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
        sampled_nodes[l][i] = candidate_vid[mid];
      }

      cout << "Nodes sampled\n";

      // layer-dependent laplacian matrix
      float interm[LADIES_SAMPLE];

      for(i=0; i<LADIES_SAMPLE; ++i) {
        interm[i] = prob[sampled_nodes[l][i]];
      }

      uint32_t e=0, ind;
      for(i=0; i<LADIES_SAMPLE; ++i) { // first row of Q.P
        ind = sampled_nodes[l+1][i];
        sampled_offset[l][i] = e;
        for(j=0; j<LADIES_SAMPLE; ++j) {
          uint32_t val=0;
          for(uint32_t k=vertex_ptr[ind]; k<vertex_ptr[ind+1]; ++k) {
            if(edge_list[k]==sampled_nodes[l][j]) {
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
    cout << "Done doing ladies sampling for first batch with edges: " << sampled_offset[0][LADIES_SAMPLE] << "\n";
    /*for(int k=0; k<=LADIES_SAMPLE; ++k) {
      cout << "offset at k: " << k << " is " << sampled_offset[0][k] << endl; 
    }
    for(int k=0; k<=sampled_offset[0][LADIES_SAMPLE]; ++k) {
      cout << "dest at k: " << k << " is " << sampled_edge_list[0][k] << endl; 
    }*/
}

int main() {

  struct gcn_info *info = (struct gcn_info*)malloc(sizeof(struct gcn_info));


  read_adjacency_matrix(info->edge_list, info->wgt, info->vertex_ptr);

  // TODO: add later
  // static VTYPE weights[GCN_LAYERS][FEAT_LEN][FEAT_LEN];
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

  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(feat_active_core_mask, i);
  }

  int weights_load = wgt_rem_loc + FEAT_LEN*FEAT_LEN*4;
  assert(weights_load<NUM_THREADS*SCRATCH_SIZE && "weights cross the allocated scratch space");
  int wgt_req_cores = FEAT_LEN*FEAT_LEN*4/SCRATCH_SIZE;
  cout << "Required cores for weight: " << wgt_req_cores << endl;
  // TODO: round wgt to the nearest power of 2
  for(int i=WGT_CORE; i<WGT_CORE+wgt_req_cores; ++i) {
    addDest(wgt_active_core_mask, i);
  }

  int scratch_space = NUM_VERT_PER_THREAD*FEAT_LEN*4;
  assert(scratch_space<FEAT_PART_SIZE && "required scratch space is more than available partition, increase cores");
  assert(FEAT_LEN%VEC_LEN==0 && "feat_len should be a multiple of vec_len");
  assert(FEAT_LEN%16==0 && "currently only support 64-byte multiple wide type");
  assert(LADIES_SAMPLE%NUM_THREADS==0 && "Not sure otherwise how we distributed");

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

