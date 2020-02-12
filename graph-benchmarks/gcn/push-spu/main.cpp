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
#include "agg.dfg.h"
#include "mult.dfg.h"
#include "/home/vidushi/ss-stack/ss-tools/include/ss-intrin/ss_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include "assert.h"
#define VTYPE float

using namespace std;

// local - 3648
// remote - 12228 // 15285

// #define NODES 1971
#define rate 0.073
#define FEAT_LEN 64 // 128 // 64 // 256 // 128 // 256 // 64 // 256 // 16 // 16 // 32 // 16 // 32 // 64 // 16 // 32 // 64 // 32 // 16 // 128 // 64
#define VEC_LEN 16

// #define LADIES_SAMPLE 128 // 32 // 36 // 32 // 16 // 256 // 16 // 64 // 128 // 512 // 256 // 512
#define LADIES_EDGES 2048

// this should be strictly less than C
// #define NUM_THREADS 8 // 8 // 8 // 4 // 2 // 8
#define NUM_VERT_PER_THREAD (LADIES_SAMPLE/NUM_THREADS)
#define SCRATCH_SIZE 32768 // 16384
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
  uint32_t feature_map[2][V][FEAT_LEN];
  // uint32_t edge_list[E]; 
  // VTYPE wgt[E];
  // uint32_t vertex_ptr[V+1];
  // vector<uint32_t> sampled_edge_list[GCN_LAYERS];
  uint32_t sampled_edge_list[GCN_LAYERS][LADIES_EDGES];
  uint32_t sampled_offset[GCN_LAYERS][LADIES_SAMPLE+1];
 
};


const int factor = FEAT_LEN/VEC_LEN;

uint32_t sampled_nodes[GCN_LAYERS+1][LADIES_SAMPLE];
int wgt_rem_loc = WGT_CORE<<SCRATCH_BITS;
uint64_t feat_active_core_mask=0;
uint64_t wgt_active_core_mask=0;
uint64_t broadcast_mask=0;

  
int feat_part = NUM_VERT_PER_THREAD*FEAT_LEN*4; // 16384*4/NUM_THREADS; // nearest power of 2
// int feat_part = 2*NUM_VERT_PER_THREAD*FEAT_LEN*4; // 16384*4/NUM_THREADS; // nearest power of 2
// int feat_part = 16384*2/NUM_THREADS; // 2*NUM_VERT_PER_THREAD*FEAT_LEN*4; // 16384*4/NUM_THREADS; // nearest power of 2

void agg(bool profiling, long tid, uint32_t (&feature_map)[2][V][FEAT_LEN], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1]) {

  VTYPE weights[GCN_LAYERS][FEAT_LEN][FEAT_LEN];
  int num_tiles = FEAT_LEN/16; // TODO: 1 or 2 doesn't work?


  uint32_t start_col = tid*NUM_VERT_PER_THREAD;
  uint32_t end_col = (tid+1)*NUM_VERT_PER_THREAD;
  int cur_active_vert=NUM_VERT_PER_THREAD;

  int wgt_part = (FEAT_LEN*FEAT_LEN*4)/NUM_THREADS; // FIXME:should be a power of 2?
  assert(ceil(log2(wgt_part))==floor(log2(wgt_part)) && "current weight partition is not a power of 2");
  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(wgt_active_core_mask,i);
  }
  int wgt_start = SCRATCH_SIZE-wgt_part;

  for(unsigned k=start_col; k<end_col; ++k) {
    int degree = sampled_offset[0][k+1]-sampled_offset[0][k];
    if(degree==0) --cur_active_vert;
  }

  int l=0;

  // SS_DMA_SCRATCH_LOAD(&sampled_offset[l][start_col+1], 4, 4, NUM_VERT_PER_THREAD-1, 0);
  // SS_WAIT_ALL();
  // SS_WAIT_SCR_WR();

  if(profiling) begin_roi();

#if AGG==1
  // TODO: remove dst_id_in (don't want mem->port to become bottleneck)
  // this is to calculate the degree (offset[start_col+1]-offset[start_col]
  SS_CONST(P_agg_offset_list0,sampled_offset[l][start_col],1);
  SS_ADD_PORT(P_agg_offset_list0);
  SS_DMA_READ(&sampled_offset[l][start_col+1], 4, 4, NUM_VERT_PER_THREAD-1, P_agg_offset_list1);
  // SS_SCR_PORT_STREAM(0, 4, 4, NUM_VERT_PER_THREAD-1, P_agg_offset_list1); //
  // TODO: if this is a problem, it should be moved to scratch (not sure why
  // giving error)
  SS_CONST(P_agg_offset_list1,sampled_offset[l][end_col],1);

  // accessing the dst_id (src, src+1, src+2)
  SS_CONFIG_INDIRECT(T32,T32,4,1); // multiplier for offset
  SS_INDIRECT_2D(P_agg_start_ind, &sampled_edge_list[l][0], NUM_VERT_PER_THREAD, 4, 4, P_agg_row_size1, P_agg_dst_id_in); // dest_id

  SS_DMA_READ(&feature_map[0][0][0], 4, 4, cur_active_vert*FEAT_LEN, P_agg_feat_in);

  // val_num is also mult for the address..
  SS_CONFIG_MEM_MAP(feat_part,feat_active_core_mask,0);
  SS_CONFIG_ATOMIC_SCR_OP(T32, T32, T32, FEAT_LEN, P_agg_row_size2, 1);
  SS_ATOMIC_SCR_OP(P_agg_dst_id_out, P_agg_feat_out, 0, NUM_VERT_PER_THREAD, 0); // num_edges, NUM_VERT_PER_THREAD*FEAT_LEN

  SS_WAIT_ALL();
  SS_GLOBAL_WAIT(NUM_THREADS);

#endif

#if MULT==1

  SS_CONFIG(mult_config, mult_size);

  // reduction from feat_len/16 to 1 (serialize or general?)
  // SS_RECURRENCE(P_mult_A, P_mult_red, num_tiles*NUM_VERT_PER_THREAD*FEAT_LEN);
  SS_2D_CONST(P_mult_const, 0, num_tiles-1, 1, 1, NUM_VERT_PER_THREAD*FEAT_LEN);
  SS_SCR_PORT_STREAM(0, 0, 4*16*num_tiles*NUM_VERT_PER_THREAD, FEAT_LEN, P_mult_agg_feat);
  for(int f=0; f<FEAT_LEN; ++f) { // dim of weights

    // should be broadcast from memory to each core's scratchpad
    // then wait on a broadcast?
    if(tid==0) { // f%NUM_THREADS) {
      SS_DMA_READ(&weights[l][f][0], 4, 4, FEAT_LEN, P_IND_1);
      SS_REM_PORT(P_IND_1, FEAT_LEN*4, broadcast_mask, MEM_SCR_PORT);
    }
    SS_SCR_WRITE(MEM_SCR_PORT, FEAT_LEN*4, 0);
    SS_WAIT_SCR_WR();
    
    // column-major form (single column of weight)
    SS_SCR_PORT_STREAM(0, 0, 4*16*num_tiles, NUM_VERT_PER_THREAD, P_mult_weights);
  }
  SS_SCR_WRITE(P_mult_sum, FEAT_LEN*4*NUM_VERT_PER_THREAD, 0);

  SS_WAIT_ALL();
  SS_GLOBAL_WAIT(NUM_THREADS);


#endif

  if(profiling) {
    end_roi();
    sb_stats();
  }

}

void gcn(bool profiling, long tid, uint32_t (&feature_map)[2][V][FEAT_LEN], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES], uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1]) {

  // TODO: use the correct value of feature map
  // TODO: should be 1 after correct reduction
  // should be read from scratch..
  // static uint32_t feature_map[2][V][FEAT_LEN];
  int wgt_part = (FEAT_LEN*FEAT_LEN*4)/NUM_THREADS; // FIXME:should be a power of 2?
  assert(ceil(log2(wgt_part))==floor(log2(wgt_part)) && "current weight partition is not a power of 2");
  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(wgt_active_core_mask,i);
  }
  // FIXME: not sure about this error..
  int wgt_start = 0; // SCRATCH_SIZE-wgt_part;

  uint32_t weights[GCN_LAYERS][FEAT_LEN][FEAT_LEN];
  
  uint32_t start_col = tid*NUM_VERT_PER_THREAD;
  uint32_t end_col = (tid+1)*NUM_VERT_PER_THREAD;
  int cur_active_vert=NUM_VERT_PER_THREAD;
  int num_tiles = FEAT_LEN/16;

  int nodes_in_core[NUM_THREADS];
  int nodes_done=0;
  int nodes_possible = feat_part/(FEAT_LEN*4);
  for(int c=0; c<NUM_THREADS && nodes_done<LADIES_SAMPLE; ++c) {
    nodes_in_core[c] = std::min(LADIES_SAMPLE-nodes_done, nodes_possible);
    nodes_done += nodes_in_core[c];
  }
  // cout << "tid: " << tid << " cur active vertex: " << cur_active_vert << endl;
  if(profiling) begin_roi();
  for(int b=0; b<NUM_BATCH; ++b) {
    for(uint32_t layer=0; layer<GCN_LAYERS; ++layer) {

      uint32_t l=layer; // 0; // layer;

      // TODO: this should not be required, wgt matrix won't be in sparse format
      cur_active_vert=NUM_VERT_PER_THREAD;
      for(unsigned k=start_col; k<end_col; ++k) {
        int degree = sampled_offset[l][k+1]-sampled_offset[l][k];
        if(degree==0) --cur_active_vert;
      }

      // Write feature map to scratchpad
      // SS_DMA_READ(&feature_map[0][start_col][0], 4, 4, LADIES_SAMPLE*FEAT_LEN, MEM_SCR_PORT);
      // SS_CONFIG_MEM_MAP(feat_part,feat_active_core_mask,0); // where to write to scratch
      // TODO: support config mem map here, also this should write globally
      // SS_SCR_WRITE(MEM_SCR_PORT, LADIES_SAMPLE*FEAT_LEN*4, 0);
      int new_start_col = tid*nodes_possible;
      // loading the destination nodes in the scratchpad
      SS_DMA_SCRATCH_LOAD(&feature_map[0][new_start_col][0], 1, 1, feat_part, 0);
      SS_WAIT_ALL();
      SS_GLOBAL_WAIT(NUM_THREADS);
#if AGG==1
      // this is to calculate the degree (offset[start_col+1]-offset[start_col]
      // TODO: can I somwhow avoid this overhead or prioritize this somehow?
      SS_CONST(P_gcn_offset_list0,sampled_offset[l][start_col],1);
      SS_ADD_PORT(P_gcn_offset_list0);
      SS_DMA_READ(&sampled_offset[l][start_col+1], 4, 4, NUM_VERT_PER_THREAD-1, P_gcn_offset_list1);
      SS_CONST(P_gcn_offset_list1,sampled_offset[l][end_col],1);

      // accessing the dst_id (src, src+1, src+2)
      SS_CONFIG_INDIRECT(T32,T32,4,1); // multiplier for offset
      SS_INDIRECT_2D(P_gcn_start_ind, &sampled_edge_list[l][0], NUM_VERT_PER_THREAD, 4, 4, P_gcn_row_size1, P_gcn_dst_id_in); // dest_id
     
      // source feature vectors
      SS_DMA_READ(&feature_map[0][start_col][0], 4, 4, cur_active_vert*FEAT_LEN, P_gcn_feat_in);

      SS_CONFIG_MEM_MAP(feat_part,feat_active_core_mask,0);
      SS_CONFIG_ATOMIC_SCR_OP(T32, T32, T32, FEAT_LEN, P_gcn_row_size2, 1);
      SS_ATOMIC_SCR_OP(P_gcn_dst_id_out, P_gcn_feat_out, 0, NUM_VERT_PER_THREAD, 0); 
#endif
      // nodes which 1. have an incoming edge 2. belong to the current thread
      // according to the mapping
      // matrix-vector multiplication (active_vert*feat_len)
#if MULT==1
      // SS_2D_CONST(P_gcn_const, 1, FEAT_LEN-1, 0, 1, nodes_in_core[tid]);
      // SS_CONFIG_MEM_MAP(wgt_part,wgt_active_core_mask,0);
      
      // SS_CONFIG_MEM_MAP(SCRATCH_SIZE,wgt_active_core_mask,0);
      // SS_CONFIG_INDIRECT(T32, T32, 4, FEAT_LEN);
      // SS_INDIRECT_SCR(P_gcn_running_sum, wgt_start, FEAT_LEN*nodes_in_core[tid], P_gcn_weights);
      SS_CONST(P_gcn_weights, 1, FEAT_LEN*FEAT_LEN*nodes_in_core[tid]);
      SS_2D_CONST(P_gcn_new_const, 0, factor-1, 1, 1, nodes_in_core[tid]*FEAT_LEN);
      SS_SCR_WRITE(P_gcn_sum, FEAT_LEN*4*nodes_in_core[tid], 0);

      for(int k=0; k<nodes_in_core[tid]; ++k) {
        uint64_t x;
        SS_RECV(P_gcn_trigger, x); // we should wait on receive here... (how to wait for last update to be written?)
        SS_SCR_PORT_STREAM(0, 0, 4*FEAT_LEN, FEAT_LEN, P_gcn_aggfeat);
      }

      // SS_DMA_READ(&weights[l][0][0], 0, 4*FEAT_LEN*FEAT_LEN, nodes_in_core[tid], P_gcn_weights);
      // SS_DMA_READ(&weights[l][0][0], 0, 4*FEAT_LEN, nodes_in_core[tid], P_gcn_weights);
      // // SS_CONST(P_gcn_weights, 1, FEAT_LEN*nodes_in_core[tid]);
      // SS_2D_CONST(P_gcn_new_const, 0, factor-1, 1, 1, nodes_in_core[tid]);
      // SS_SCR_WRITE(P_gcn_sum, 4*nodes_in_core[tid], 0);
      
#endif
      SS_WAIT_ALL();
      SS_GLOBAL_WAIT(NUM_THREADS);
    }
  }
  if(profiling) {
    end_roi();
    sb_stats();
  }
}

void read_sampling_arrays(uint32_t (&sampled_offset)[GCN_LAYERS][LADIES_SAMPLE+1], uint32_t (&sampled_edge_list)[GCN_LAYERS][LADIES_EDGES]) {
  char linetoread[50000];

  cout << "Starting reading sampled nodes file\n";
  FILE* node_map = fopen("sampled_nodes.txt", "r");
  int l=0; 
  while(fgets(linetoread, 50000, node_map) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    // cout << "line read: " << raw << endl;
    string ignore;
    int i=-1;
    while (getline(iss, ignore, ' ')) {
      sampled_nodes[l][++i] = atoi(ignore.c_str());
    } 
   ++l;
  }
  fclose(node_map);

  cout << "Starting reading adj matrix file\n";
  FILE* adj_new = fopen("adj_mat.txt", "r");

  int row=0;
  while(fgets(linetoread, 50000, adj_new) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    string ignore;
    int i=-1;
    l=row/2;
    while (getline(iss, ignore, ' ')) {
      if(row%2==0) { // vertex_ptr
        cout << "Allotted offset of index: " << i << endl;
        sampled_offset[l][++i] = atoi(ignore.c_str());
      } else { // edge list
        sampled_edge_list[l][++i] = atoi(ignore.c_str());
        assert(sampled_edge_list[l][i]<LADIES_SAMPLE);
        cout << "Allotted edge of index: " << i << endl;
      }
    } 
    if(row%2==0) cout << "Total edges in layer l: " << l << " is: " << sampled_offset[l][LADIES_SAMPLE] << endl;
    ++row;
  }
  fclose(adj_new);
  cout << "Done reading all files\n";

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
    iss >> src >> dst >> x;
    // iss >> src >> dst;
    // src = src-1;
    // dst = dst-1;

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
 
#if SYNC==0
  SS_CONFIG(gcn_config, gcn_size);
  SS_ATOMIC_DFG_CONFIG(P_gcn_scr_addr_in, P_gcn_scr_val_in, P_gcn_scr_out);
#else
  SS_CONFIG(agg_config, agg_size);
  SS_ATOMIC_DFG_CONFIG(P_agg_scr_addr_in, P_agg_scr_val_in, P_agg_scr_out);
#endif
  SS_GLOBAL_WAIT(NUM_THREADS);

  // Synchronization pouint32_t
  uint32_t rc = pthread_barrier_wait(&barr);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
  {
    printf("Could not wait on barrier\n");
  }

#if SYNC==0
  assert(FEAT_LEN*4==4*64 && "feat len doesn't match with vec width encoded in gcn");
  // gcn(false, tid, ((struct gcn_info*)info)->feature_map, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
  // SS_CONFIG(gcn_config, gcn_size);
  // SS_ATOMIC_DFG_CONFIG(P_gcn_scr_addr_in, P_gcn_scr_val_in, P_gcn_scr_out);
  gcn(true, tid, ((struct gcn_info*)info)->feature_map, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
#else
  agg(false, tid, ((struct gcn_info*)info)->feature_map, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
#if AGG==1 // initialize all ports
  SS_CONFIG(agg_config, agg_size);
  SS_ATOMIC_DFG_CONFIG(P_agg_scr_addr_in, P_agg_scr_val_in, P_agg_scr_out);
  SS_GLOBAL_WAIT(NUM_THREADS);
#endif
  agg(true, tid, ((struct gcn_info*)info)->feature_map, ((struct gcn_info*)info)->sampled_edge_list, ((struct gcn_info*)info)->sampled_offset);
#endif

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

    for(int l=0; l<GCN_LAYERS; ++l) {
      cout << "Done doing ladies sampling for first batch with edges: " << sampled_offset[l][LADIES_SAMPLE] << "\n";
    }
    /*for(int k=0; k<=LADIES_SAMPLE; ++k) {
      cout << "offset at k: " << k << " is " << sampled_offset[0][k] << endl; 
    }
    for(int k=0; k<=sampled_offset[0][LADIES_SAMPLE]; ++k) {
      cout << "dest at k: " << k << " is " << sampled_edge_list[0][k] << endl; 
    }*/
}


// in-degree
void preprocess_ld_balance(int (&indegree)[LADIES_SAMPLE]) {
  int cur_average=0;
  int vert_done=0;
  vector<int> cur_degree;
  cur_degree.resize(NUM_THREADS, 0);
  vector<int> vertices_in_core[NUM_THREADS];
  while(vert_done<LADIES_SAMPLE) {
    // allocated vertices to all cores
    bool entered=false;
    for(int i=0; i<NUM_THREADS && vert_done<LADIES_SAMPLE; ++i) {
      while(cur_degree[i] < cur_average && vert_done<LADIES_SAMPLE) {
        entered=true;
        // map vertices to core i
        vertices_in_core[i].push_back(vert_done);
        cur_degree[i] += indegree[vert_done];
        cout << "vertex pushed: " << vert_done << " and degree: " << cur_degree[i] << endl;
        vert_done++;
      }
    }
    // update cur average here
    cur_average=0;
    for(int i=0; i<NUM_THREADS; ++i) {
      cur_average += cur_degree[i];
    }
    cur_average /= NUM_THREADS;
  }

  // print the mapping (FIXME: BUT WE CANNOT MAP AT SAME CORE)
  for(int i=0; i<NUM_THREADS; ++i) {
    cout << "core: " << i << " ";
    for(int j=0; vertices_in_core[i].size(); ++j) {
      cout << vertices_in_core[i][j] << " ";
    }
    cout << endl;
  }
  exit(0);
}


int main() {

  struct gcn_info *info = (struct gcn_info*)malloc(sizeof(struct gcn_info));


#if FULL_TRAINING==1
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
#endif

  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(broadcast_mask, i);
  }
  
  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(feat_active_core_mask, i);
  }

  int weights_load = wgt_rem_loc + FEAT_LEN*FEAT_LEN*4;
  assert(weights_load<NUM_THREADS*SCRATCH_SIZE && "weights cross the allocated scratch space");
  int wgt_req_cores = FEAT_LEN*FEAT_LEN*4/SCRATCH_SIZE;
  wgt_req_cores = std::max(1, wgt_req_cores);
  cout << "Required cores for weight: " << wgt_req_cores << endl;
  // TODO: round wgt to the nearest power of 2
  for(int i=WGT_CORE; i<WGT_CORE+wgt_req_cores; ++i) {
    addDest(wgt_active_core_mask, i);
  }

  int scratch_space = NUM_VERT_PER_THREAD*FEAT_LEN*4;
  assert(scratch_space<FEAT_PART_SIZE && "required scratch space is more than available partition, increase cores");
  assert(FEAT_LEN%VEC_LEN==0 && "feat_len should be a multiple of vec_len");
  assert(FEAT_LEN%16==0 && "currently only support 64-byte multiple wide type, also tile factor should be a multiple");
  assert(LADIES_SAMPLE%NUM_THREADS==0 && "Not sure otherwise how we distributed");

#if FULL_TRAINING==1
  perform_sampling(info->edge_list, info->wgt, info->vertex_ptr, info->sampled_offset, info->sampled_edge_list);
  cout << "Sampling done\n";
#else
  read_sampling_arrays(info->sampled_offset, info->sampled_edge_list);
#endif
  /*for(int k=0; k<LADIES_SAMPLE; ++k) {
    cout << "offset at k: " << k << " is: " << info->sampled_offset[0][k] << endl;
  }*/
#if SYNC==0
  // TODO: doesn't work for multiple layers (move to gcn)
  int num_inc_nodes[LADIES_SAMPLE] = {0};
  for(int i=0; i<LADIES_SAMPLE; ++i) {
    for(int j=info->sampled_offset[0][i]; j<info->sampled_offset[0][i+1]; ++j) {
      num_inc_nodes[info->sampled_edge_list[0][j]]++;
    }
  }
  for(int i=0; i<LADIES_SAMPLE; ++i) {
    for(int j=0; j<FEAT_LEN; ++j) {
      info->feature_map[0][i][j]=100;
    }
  }
  for(int i=0; i<LADIES_SAMPLE; ++i) {
    info->feature_map[0][i][0]=num_inc_nodes[i];
    cout << "Number of incoming nodes: " << num_inc_nodes[i] << endl;
  }
  for(unsigned k=0; k<LADIES_SAMPLE; ++k) {
    int degree = info->sampled_offset[0][k+1]-info->sampled_offset[0][k];
    // cout << "Degree of k: " << k << " is: " << degree << endl;
  }
  int nodes_possible = feat_part/(FEAT_LEN*4);
  int nodes_done=0;
  int nodes_in_core[NUM_THREADS];
  for(int c=0; c<NUM_THREADS && nodes_done<LADIES_SAMPLE; ++c) {
    nodes_in_core[c] = std::min(LADIES_SAMPLE-nodes_done, nodes_possible);
    nodes_done += nodes_in_core[c];
    cout << "Nodes in core c: " << nodes_in_core[c] << " ";
  }
  cout << "Total nodes done: " << nodes_done << endl;
#endif
#if LOAD_BALANCE==1
  preprocess_ld_balance(num_inc_nodes);
#endif
  /*
  for(int i=0; i<LADIES_SAMPLE; ++i) {
    int degree = info->sampled_offset[0][i+1]-info->sampled_offset[0][i]; 
    cout << "Degree of node i: " << i << " degree: "  << degree << endl;
  }*/
  /*for(int k=0; k<info->sampled_offset[0][LADIES_SAMPLE]; ++k) {
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


