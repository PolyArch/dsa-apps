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
  uint64_t nodeType; 
  double vr; 
  double dr; 
  uint64_t child0;
  uint64_t child1;
  uint64_t flag;
};



void forwardPropagation(tree* circuit, uint64_t nodes_at_level[d2-d1], uint64_t levels, uint64_t index){
  int n_times = 0;
  // int offset = (int)(&circuit.flag[0]-&circuit.vr[0]);
  int offset = index;
  // int d = levels-1;
  int d = levels;
  std::cout << "levels are: " << levels << "\n";
  // int d = d2-d1-1;
  int i=index-nodes_at_level[d]; // need starting index
  // printf("number of nodes per cgra are: %d and level we start from %d\n", index, d);
  // std::cout << sizeof(tree) << "\n";
  /*
  for(int i=47; i<49; i++){
	std::cout << *(&circuit[47].child0+(sizeof(tree)*1*(i-47)/8)) << "\n";
  }
  */
  SB_CONFIG(fwd_prop_config,fwd_prop_size);

  for (d=d-1; d >= 0; d--) {
    // start with i
    i-=nodes_at_level[d-1];
    n_times = nodes_at_level[d];
	// std::cout << i << " " << n_times << "\n";
   
    // std::cout << "number of elements at the level: " << d << " are " << n_times << " with the starting index: " << i << std::endl;
    SB_DMA_READ(&circuit[i].nodeType, sizeof(tree), 8, n_times, P_fwd_prop_nodeType);
    // SB_DMA_READ(&circuit[i].nodeType, 8, 8, n_times, P_fwd_prop_nodeType);
    // SB_DMA_READ(&circuit[i].child0, 8, 8, n_times, P_IND_1);
    SB_DMA_READ(&circuit[i].child0, sizeof(tree), 8, n_times, P_IND_1);
    SB_DMA_READ(&circuit[i].child1, sizeof(tree), 8, n_times, P_IND_2);

	// SB_CONFIG_INDIRECT1(T64,T64,1,&circuit.flag[0]-&circuit.vr[0]);
	// SB_CONFIG_INDIRECT1(T64,T64,1,offset);
	// SB_CONFIG_INDIRECT1(T64,T64,(offset*6),offset);
	// SB_CONFIG_INDIRECT1(T64,T64,1,1);
	SB_CONFIG_INDIRECT1(T64,T64,sizeof(struct tree)/8,4*sizeof(uint64_t)/8);
    SB_INDIRECT(P_IND_1, &circuit[i].vr, n_times, P_fwd_prop_c1vf);

	// SB_CONFIG_INDIRECT1(T64,T64,1,offset);
	// SB_CONFIG_INDIRECT1(T64,T64,(offset*6),offset);
	// SB_CONFIG_INDIRECT1(T64,T64,1,1);
	SB_CONFIG_INDIRECT1(T64,T64,sizeof(struct tree)/8,4*sizeof(uint64_t)/8);
    SB_INDIRECT(P_IND_2, &circuit[i].vr, n_times, P_fwd_prop_c2vf);

    // SB_CONFIG_INDIRECT(T64,T64,1);
	// // Oh it will read like vector!
    // SB_INDIRECT(P_IND_1, &circuit.vr[i], n_times, P_fwd_prop_c1vr);
    // SB_INDIRECT(P_IND_1, &circuit.flag[i], n_times, P_fwd_prop_c1vr);

	// SB_CONFIG_INDIRECT(T64,T64,1);
	// // Oh it will read like vector!
    // SB_INDIRECT(P_IND_2, &circuit.vr[i], n_times, P_fwd_prop_c2vr);
    // SB_INDIRECT(P_IND_2, &circuit.flag[i], n_times, P_fwd_prop_c2vr);


	SB_DMA_WRITE(P_fwd_prop_vr, sizeof(tree), 8, n_times, &circuit[i].vr);
	SB_DMA_WRITE(P_fwd_prop_flag, sizeof(tree), 8, n_times, &circuit[i].flag);

    // SB_WAIT_COMPUTE();
    SB_WAIT_ALL();
    // i+=nodes_at_level[d];
  }
  // std::cout << "Number of computations done: " << i << std::endl;
}

int main() {
  FILE *ac_file;
  char lineToRead[MAX_SIZE]; 
  uint64_t nodes_at_level[d2-d1];
  uint64_t cum_nodes_at_level[d2-d1];
  int cur_level=0;
  struct tree* arith_ckt;
  uint64_t index = 0;
  uint64_t *a, *b, *c, *d;

  arith_ckt = (struct tree*)malloc(sizeof(struct tree) * MAX_SIZE);
  a = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  b = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  c = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  d = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  
  ac_file = fopen("input.data", "r");

  printf("Started reading file!\n");

  while (fgets(lineToRead, MAX_SIZE, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    // std::string level[7];
    char op;
    uint64_t n;
    char ignore;
    // iss >> level >> ignore >> ignore >> n >> ignore >> op >> ignore;
	// std::cout << iss.str() << "\n";
	// iss >> level >> m >> ignore >> n >> op;
	iss >> level >> n >> op;
	
	// std::cout << level << std::endl;
	// std::cout << n << std::endl;
	// std::cout << op << std::endl;
	//printf("n %ld op %c\n",n,op);
    
    nodes_at_level[cur_level] = n;
    if(cur_level==0){
      cum_nodes_at_level[cur_level] = n;
    } else {
      cum_nodes_at_level[cur_level] = (n+cum_nodes_at_level[cur_level-1]);
    }
    for (uint64_t i = 0; i < n * 2; i+=2) {
        // iss >> a[index] >> ignore >> b[index] >> ignore >> c[index] >> ignore >> d[index];
        // iss >> a[index] >> ignore >> b[index] >> ignore >> c[index] >> ignore >> d[index];
        iss >> a[index] >> ignore >> b[index] >> c[index] >> ignore >> d[index];
		// std::cout << a[index] << " " << b[index] << " " << c[index] << " " << d[index] << "\n";
        if(op=='*'){
          arith_ckt[index].nodeType=1;
        } else {
          arith_ckt[index].nodeType=0;
        }
        arith_ckt[index].vr = 1 + rand()/RAND_MAX; // What is uint? I want double, right?
        arith_ckt[index].dr = 1 + rand()/RAND_MAX;
        arith_ckt[index].flag = rand()%2;
        index++;
    }
    cur_level++;
  }

  uint64_t child1_ind, child2_ind;
  for(uint64_t i=0; i<index; i++){
    child1_ind = cum_nodes_at_level[a[i]-1]+b[i];
    child2_ind = cum_nodes_at_level[c[i]-1]+d[i];
    arith_ckt[i].child0 = child1_ind;
    arith_ckt[i].child1 = child2_ind;
	// printf("i: %ld child1: %ld child2: %ld\n",i,arith_ckt[i].child0,arith_ckt[i].child1);
  }

  // int no_nodes_per_cgra = index/C + 1;
  // int no_nodes_per_cgra = index/C;
  // if(index<2192)
  //  std::cout << "increase the input size\n";
  // d2 = 25, 15
  // int no_nodes_per_cgra = 2192;
  // d2 = 26, 17
  // int no_nodes_per_cgra = 3340;
  // d2=24, 14
  int no_nodes_per_cgra = 1484;
  // int no_nodes_per_cgra = index;
  
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

  forwardPropagation(arith_ckt, nodes_at_level, levels, index);
  begin_roi();
  // backPropagation(arith_ckt, nodes_at_level, levels);
  forwardPropagation(arith_ckt, nodes_at_level, levels, index);
  // forwardPropagation(arith_ckt, nodes_at_level, levels, no_nodes_per_cgra);
  end_roi();
  sb_stats();
  printf("Forward propagation done!\n");
  // printf("Backpropagation done!\n");
  return 0;
}


/*
void backPropagation(tree circuit, int nodes_at_level[d2-d1], int levels){
  // 2 parents, overwrite the value? how is this done? but should not be the
  // problem in my example
  int n_times = 0;
  int i=0;
  // int levels = d2-d1-1;
  SB_CONFIG(test_config,test_size);

  // for (int d = 0; d < levels && i<no_nodes_per_cgra; d++) {
  for (int d = 0; d < levels-1; d++) {
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
*/


