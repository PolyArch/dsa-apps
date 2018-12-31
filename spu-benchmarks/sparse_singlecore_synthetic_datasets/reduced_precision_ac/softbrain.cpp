#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "fwd_prop.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include <inttypes.h>
#include <sstream>

#define C 64
// #define C 1
#define MAX_SIZE 100000
#define fused_const (1 | (1 & 0xFFFFFFFF00000000) << 32)
	
struct tree {
  uint32_t nodeType [2]; 
  float vr [2]; 
  float dr [2]; 
  uint32_t child0 [2];
  uint32_t child1 [2];
  uint32_t flag [2];
};

void forwardPropagation(tree* circuit, uint32_t nodes_at_level[d2-d1], uint32_t levels, uint32_t index, int no_nodes_per_cgra){
  int n_times = 0;
  // int offset = (int)(&circuit.flag[0]-&circuit.vr[0]);
  // int offset = index;
  // int d = levels-1;
  int d = levels-1;
  // std::cout << "levels are: " << levels << "\n";
  // int d = d2-d1-1;
  int i=index-nodes_at_level[d]; // need starting index
  // printf("number of nodes per cgra are: %d and level we start from %d\n", no_nodes_per_cgra, d);
  no_nodes_per_cgra /= 2;
  // std::cout << sizeof(tree) << "\n";

  int vr_offset = 0;
  SS_DMA_SCRATCH_LOAD(&circuit[0].vr, 8, 8, no_nodes_per_cgra/2, vr_offset);
  SS_WAIT_SCR_WR();


  for (d=d-1; d >= 0; d--) {
    // start with i
    i-=(nodes_at_level[d-1]);
    n_times = nodes_at_level[d];
	// vectorization width of 2
	n_times = n_times/2;
	//std::cout << "depth: " << d << " " << i << " num_times: " << n_times << "\n";
   
    // std::cout << "number of elements at the level: " << d << " are " << n_times << " with the starting index: " << i << std::endl;
    SS_DMA_READ(&circuit[i].nodeType, sizeof(tree), 8, n_times, P_fwd_prop_nodeType);
    SS_DMA_READ(&circuit[i].child0, sizeof(tree), 8, n_times, P_IND_1);
    SS_DMA_READ(&circuit[i].child1, sizeof(tree), 8, n_times, P_IND_2);
	SS_CONST(P_fwd_prop_const, fused_const, n_times);

	// SS_CONFIG_INDIRECT1(T64,T64,sizeof(struct tree),4*sizeof(uint32_t));
	// SS_CONFIG_INDIRECT1(T32,T32,sizeof(struct tree),4*sizeof(uint32_t));
	// SS_CONFIG_INDIRECT1(T32, T64, sizeof(tree), 2*sizeof(uint32_t));
	// should extract 2 32-bit value per index
	SS_CONFIG_INDIRECT1(T32, T32, sizeof(tree), 2*sizeof(uint32_t));
    // SS_INDIRECT_SCR(P_IND_1, vr_offset, n_times, P_fwd_prop_c1vf);
    SS_INDIRECT_SCR(P_IND_1, vr_offset, 2*n_times, P_fwd_prop_c1vf);

	// SS_CONFIG_INDIRECT1(T64,T64,sizeof(struct tree),4*sizeof(uint32_t));
	// SS_CONFIG_INDIRECT1(T32,T32,sizeof(struct tree),4*sizeof(uint32_t));
	SS_CONFIG_INDIRECT1(T32, T32, sizeof(tree), 2*sizeof(uint32_t));
    SS_INDIRECT_SCR(P_IND_2, vr_offset, 2*n_times, P_fwd_prop_c2vf);

	SS_DMA_WRITE(P_fwd_prop_vr, sizeof(tree), 8, n_times, &circuit[i].vr);
	SS_DMA_WRITE(P_fwd_prop_flag, sizeof(tree), 8, n_times, &circuit[i].flag);

    SS_WAIT_ALL();
  }
  // std::cout << "Number of computations done: " << i << std::endl;
}

int main() {
  FILE *ac_file;
  char lineToRead[MAX_SIZE]; 
  uint32_t nodes_at_level[d2-d1];
  uint32_t cum_nodes_at_level[d2-d1];
  int cur_level=0;
  struct tree* arith_ckt;
  uint32_t index = 0;
  uint32_t *a, *b, *c, *d;

  arith_ckt = (struct tree*)malloc(sizeof(struct tree) * MAX_SIZE);
  a = (uint32_t*)malloc(sizeof(uint32_t) * MAX_SIZE);
  b = (uint32_t*)malloc(sizeof(uint32_t) * MAX_SIZE);
  c = (uint32_t*)malloc(sizeof(uint32_t) * MAX_SIZE);
  d = (uint32_t*)malloc(sizeof(uint32_t) * MAX_SIZE);
  
  ac_file = fopen("input.data", "r");

  double x,y;
  printf("Started reading file!\n");
  int count_nt=-1;
  int count_vr=-1;
  int count_dr=-1;
  int count_flag=-1;
  int count_index=0;

  while (fgets(lineToRead, MAX_SIZE, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    uint32_t n;
    char ignore;
	iss >> level >> n >> op;
   
    nodes_at_level[cur_level] = n;
    if(cur_level==0){
      cum_nodes_at_level[cur_level] = n;
    } else {
      cum_nodes_at_level[cur_level] = (n+cum_nodes_at_level[cur_level-1]);
    }
	count_index=0;
	// something has to be done with odd nodes, think carefully!
    for (uint32_t i = 0; i < n * 2; i+=2) {
	  int ind = (2*index) + (count_index%2);
	  iss >> a[ind] >> ignore >> b[ind] >> c[ind] >> ignore >> d[ind];
      // iss >> a[2*index+(count_index%2)] >> ignore >> b[2*index+(count_index%2)] >> c[2*index+(count_index%2)] >> ignore >> d[2*index+(count_index%2)];
	  // std::cout << a[ind] << " " << b[ind] << " " << c[ind] << " " << d[ind] << "\n";
	  // std::cout << a[index] << " " << b[index] << " " << c[index] << " " << d[index] << "\n";
      if(op=='*'){
        // arith_ckt[index].nodeType=1;
        arith_ckt[index].nodeType[(++count_nt)%2]=1;
      } else {
        // arith_ckt[index].nodeType=0;
        arith_ckt[index].nodeType[(++count_nt)%2]=0;
      }
	  x = 1 + rand()/RAND_MAX;
	  y = 1 + rand()/RAND_MAX;
	  // arith_ckt[index].vr = x * (1<<8);
	  arith_ckt[index].vr[(++count_vr%2)] = x * (1<<8);
	  arith_ckt[index].dr[(++count_dr%2)] = y * (1<<8);
	  // arith_ckt[index].dr = y * (1<<8);
      // arith_ckt[index].vr = 1 + rand()/RAND_MAX; // What is uint? I want float, right?
      // arith_ckt[index].dr = 1 + rand()/RAND_MAX;
      arith_ckt[index].flag[(++count_flag%2)] = rand()%2;
	  count_index++;
	  if(count_index%2==0){
        index++;
	  }
    }
    cur_level++;
  }

  int count_c0=-1;
  int count_c1=-1;
  count_index=0;
  std::cout << index << "\n";

  uint32_t child1_ind, child2_ind;
  // TOTAL NUMBER OF VALUES IS NOW 2*index
  for(uint32_t i=0; i<index-1; count_index++){
	// std::cout << count_index << std::endl;
	// std::cout << i << std::endl;
	if(count_index%2==0 && count_index!=0){
	  i+=1;
	}
	int ind = (2*i) + (count_index%2);
	child1_ind = cum_nodes_at_level[a[ind]-1]+b[ind];
	child2_ind = cum_nodes_at_level[c[ind]-1]+d[ind];
    // child1_ind = cum_nodes_at_level[a[2*i+(count_index%2)]-1]+b[2*i+(count_index%2)];
    // child2_ind = cum_nodes_at_level[c[2*i+(count_index%2)]-1]+d[2*i+(count_index%2)];
    arith_ckt[i].child0[(++count_c0%2)] = child1_ind;
    arith_ckt[i].child1[(++count_c1%2)] = child2_ind;
	//std::cout << "child1: " << arith_ckt[i].child0[count_c0] << " child2: " << arith_ckt[i].child1[count_c1] << "\n";
	// std::cout << "child1: " << child1_ind << " child2: " << child2_ind << "\n";

	// printf("i: %ld child1: %ld child2: %ld\n",i,arith_ckt[i].child0,arith_ckt[i].child1);
  }

  // int no_nodes_per_cgra = index/C + 1;
  // int no_nodes_per_cgra = index/C;
  // if(index<2192)
  //  std::cout << "increase the input size\n";
  // d2 = 25, 15
  // int no_nodes_per_cgra = 2192;
  // int no_nodes_per_cgra = 2000;
  int no_nodes_per_cgra = 2*index-2;
  // d2 = 26, 17
  // int no_nodes_per_cgra = 3340;
  // d2=24, 14
  // int no_nodes_per_cgra = 1484;
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

  SS_CONFIG(fwd_prop_config,fwd_prop_size);

  forwardPropagation(arith_ckt, nodes_at_level, levels, index, no_nodes_per_cgra);
  begin_roi();
  // backPropagation(arith_ckt, nodes_at_level, levels);
  forwardPropagation(arith_ckt, nodes_at_level, levels, index, no_nodes_per_cgra);
  // forwardPropagation(arith_ckt, nodes_at_level, levels, no_nodes_per_cgra);
  end_roi();
  sb_stats();
  printf("Forward propagation done!\n");
  // printf("Backpropagation done!\n");
  return 0;
}
