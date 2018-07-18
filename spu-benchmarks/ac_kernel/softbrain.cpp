#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "test.dfg.h"
#include "fwd_prop.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>

#define C 64
// #define C 1
#define MAX_SIZE 100000

struct tree {
  uint64_t* nodeType; 
  // uint64_t* index;
  double* vr; 
  double* dr; 
  uint64_t* child0;
  uint64_t* child1;
  uint64_t* flag;
};

void forwardPropagation(tree circuit, int nodes_at_level[d2-d1], int levels, int index){
  // 2 parents, overwrite the value? how is this done? but should not be the
  // problem in my example
  int n_times = 0;
  // int offset = (int)(&circuit.flag[0]-&circuit.vr[0]);
  int offset = index;
  // int d = levels-1;
  int d = levels-2;
  // int levels = d2-d1-1;
  SB_CONFIG(fwd_prop_config,fwd_prop_size);

  // int i=0; // need to check this as well!
  int i=index-nodes_at_level[d];
  // for (d = levels-1; d >= 0; d--) {
  for (d = levels-2; d >= 0; d--) {
    // start with i
    n_times = nodes_at_level[d];
   
    // std::cout << "number of elements at the level: " << n_times << std::endl;
    SB_DMA_READ(&circuit.nodeType[i], 8, 8, n_times, P_fwd_prop_nodeType);
    SB_DMA_READ(&circuit.child0[i], 8, 8, n_times, P_IND_1);
    SB_DMA_READ(&circuit.child1[i], 8, 8, n_times, P_IND_2);

	// SB_CONFIG_INDIRECT1(T64,T64,1,&circuit.flag[0]-&circuit.vr[0]);
	SB_CONFIG_INDIRECT1(T64,T64,1,offset);
    SB_INDIRECT(P_IND_1, &circuit.vr[i], n_times, P_fwd_prop_c1vf);

	SB_CONFIG_INDIRECT1(T64,T64,1,offset);
    SB_INDIRECT(P_IND_2, &circuit.vr[i], n_times, P_fwd_prop_c2vf);

    // SB_CONFIG_INDIRECT(T64,T64,1);
	// // Oh it will read like vector!
    // SB_INDIRECT(P_IND_1, &circuit.vr[i], n_times, P_fwd_prop_c1vr);
    // SB_INDIRECT(P_IND_1, &circuit.flag[i], n_times, P_fwd_prop_c1vr);

	// SB_CONFIG_INDIRECT(T64,T64,1);
	// // Oh it will read like vector!
    // SB_INDIRECT(P_IND_2, &circuit.vr[i], n_times, P_fwd_prop_c2vr);
    // SB_INDIRECT(P_IND_2, &circuit.flag[i], n_times, P_fwd_prop_c2vr);


	SB_DMA_WRITE(P_fwd_prop_vr, 8, 8, n_times, &circuit.vr[i]);
	SB_DMA_WRITE(P_fwd_prop_flag, 8, 8, n_times, &circuit.flag[i]);

    // SB_WAIT_COMPUTE();
    SB_WAIT_ALL();
    i-=nodes_at_level[d];
    // i+=nodes_at_level[d];
  }
  // std::cout << "Number of computations done: " << i << std::endl;
}


void backPropagation(tree circuit, int nodes_at_level[d2-d1], int levels){
  // 2 parents, overwrite the value? how is this done? but should not be the
  // problem in my example
  int n_times = 0;
  // int levels = d2-d1-1;
  SB_CONFIG(test_config,test_size);

  int i=0;
  // for (int d = 0; d < levels && i<no_nodes_per_cgra; d++) {
  for (int d = 0; d < levels; d++) {
    // start with i
    n_times = nodes_at_level[d];
    
    // std::cout << "number of elements at the level: " << n_times << std::endl;
    SB_DMA_READ(&circuit.nodeType[i], 8, 8, n_times, P_test_nodeType);
    SB_DMA_READ(&circuit.dr[i], 8, 8, n_times, P_test_dr);
    SB_DMA_READ(&circuit.flag[i], 8, 8, n_times, P_test_flag);
    SB_DMA_READ(&circuit.vr[i], 8, 8, n_times, P_test_vr);
    SB_DMA_READ(&circuit.child0[i], 8, 8, n_times, P_IND_DOUB0);
    SB_DMA_READ(&circuit.child1[i], 8, 8, n_times, P_IND_1);
    SB_DMA_READ(&circuit.child1[i], 8, 8, n_times,P_IND_2);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT(P_IND_DOUB0, &circuit.vr[i], n_times, P_test_c1vr);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT(P_IND_1, &circuit.vr[i], n_times, P_test_c2vr);

    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT_WR(P_IND_DOUB1, &circuit.dr[i], n_times, P_test_c0dr);
    SB_CONFIG_INDIRECT(T64,T64,1);
    SB_INDIRECT_WR(P_IND_2, &circuit.dr[i], n_times, P_test_c1dr);

    SB_WAIT_ALL();
    i+=nodes_at_level[d];
  }
  // std::cout << "Number of computations done: " << i << std::endl;
}

int main() {
  FILE *ac_file;
  char lineToRead[MAX_SIZE]; 
  int nodes_at_level[d2-d1];
  int cum_nodes_at_level[d2-d1];
  int cur_level=0;
  struct tree arith_ckt;
  int index = 0;
  int *a, *b, *c, *d;

  // circuit = (struct node**)malloc(sizeof(struct node*) * MAX_SIZE);
  arith_ckt.nodeType = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  // arith_ckt.index = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  arith_ckt.vr = (double*)malloc(sizeof(double) * MAX_SIZE);
  arith_ckt.dr = (double*)malloc(sizeof(double) * MAX_SIZE);
  arith_ckt.child0 = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  arith_ckt.child1 = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  arith_ckt.flag = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  a = (int*)malloc(sizeof(int) * MAX_SIZE);
  b = (int*)malloc(sizeof(int) * MAX_SIZE);
  c = (int*)malloc(sizeof(int) * MAX_SIZE);
  d = (int*)malloc(sizeof(int) * MAX_SIZE);
  
  ac_file = fopen("input.data", "r");

  printf("Started reading file!\n");

  while (fgets(lineToRead, MAX_SIZE, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int n;
    iss >> level >> n >> op;
    
    nodes_at_level[cur_level] = n;
    if(cur_level==0){
      cum_nodes_at_level[cur_level] = n;
    } else {
      cum_nodes_at_level[cur_level] = (n+cum_nodes_at_level[cur_level-1]);
    }
    char ignore;
    for (int i = 0; i < n * 2; i+=2) {
        iss >> a[index] >> ignore >> b[index] >> ignore >> c[index] >> ignore >> d[index];
        if(op=='*'){
          arith_ckt.nodeType[index]=1;
        } else {
          arith_ckt.nodeType[index]=0;
        }
        arith_ckt.vr[index] = 1 + rand()/RAND_MAX; // What is uint? I want double, right?
        arith_ckt.dr[index] = 1 + rand()/RAND_MAX;
        arith_ckt.flag[index] = rand()%2;
        index++;
    }
    cur_level++;
  }

  int child1_ind, child2_ind;
  for(int i=0; i<index; i++){
    child1_ind = cum_nodes_at_level[a[i]-1]+b[i];
    child2_ind = cum_nodes_at_level[c[i]-1]+d[i];
    arith_ckt.child0[i] = child1_ind;
    arith_ckt.child1[i] = child2_ind;
  }

  int no_nodes_per_cgra = index/C + 1;
  // int no_nodes_per_cgra = index/C;
  // if(index<2192)
  //     std::cout << "increase the input size\n";
  // int no_nodes_per_cgra = 2192;
  
  printf("Done reading file!\n");

  int levels=d2-d1-1;
  // turn it off just to check forward propagation quickly
  for(int i=0; i<(d2-d1-1); i++){
    if(cum_nodes_at_level[i]>no_nodes_per_cgra){
      cum_nodes_at_level[i]=no_nodes_per_cgra;
      levels=i+1;
      break;
    }
  }

  printf("Starting backpropagation with number of nodes: %d\n", index);
  printf("Number of nodes per CGRA: %d\n", no_nodes_per_cgra);

  begin_roi();
  // backPropagation(arith_ckt, nodes_at_level, levels);
  forwardPropagation(arith_ckt, nodes_at_level, levels, index);
  end_roi();
  sb_stats();
  printf("Backpropagation done!\n");
  return 0;
}
