#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "test.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

struct tree {
  uint64_t* nodeType; 
  uint64_t* index;
  double* vr; 
  double* dr; 
  uint64_t* child0;
  uint64_t* child1;
  uint64_t* flag;
};

// void backpropogation(struct node**circuit, int index){
void backpropogation(tree circuit, int index){
  printf("\t... starting backpropagation ...\n");
  index = 5;
  begin_roi();

  SB_CONFIG(test_config,test_size);


  SB_DMA_READ(&circuit.nodeType[0],8,8,index+1,P_test_nodeType);
  SB_DMA_READ(&circuit.dr[0],8,8,index+1,P_test_dr);
  SB_DMA_READ(&circuit.flag[0],8,8,index+1,P_test_flag);
  SB_DMA_READ(&circuit.vr[0],8,8,index+1,P_test_vr);
  SB_DMA_READ(&circuit.child0[0],8,8,index+1,P_IND_DOUB0);
  // SB_DMA_READ(&circuit.child1[0],8,8,index+1,P_IND_TRIP0);
  SB_DMA_READ(&circuit.child1[0],8,8,index+1,P_IND_1);
  SB_DMA_READ(&circuit.child1[0],8,8,index+1,P_IND_2);
  // SB_CONST(P_test_const,0,index+1);
  SB_CONFIG_INDIRECT(T64,T64,1);
  SB_INDIRECT(P_IND_DOUB0, &circuit.vr[0], index+1, P_test_c1vr);
  // SB_CONFIG_INDIRECT(T64,T64,1);
  // SB_INDIRECT(P_IND_TRIP0, &circuit.vr[0], index+1, P_test_c2vr);
  SB_INDIRECT(P_IND_1, &circuit.vr[0], index+1, P_test_c2vr);

  // SB_CONFIG_INDIRECT(T64,T64,1);
  SB_INDIRECT_WR(P_IND_DOUB1, &circuit.dr[0], index+1, P_test_c0dr);
  // SB_CONFIG_INDIRECT(T64,T64,1);
  // SB_INDIRECT_WR(P_IND_TRIP1, &circuit.dr[1], index+1, P_test_c1dr);
  SB_INDIRECT_WR(P_IND_2, &circuit.dr[1], index+1, P_test_c1dr);
  // what to do with the values at trip2? can i use both trip and double at the
  // same time?

  SB_WAIT_ALL();
  end_roi();
  sb_stats();
  
}

int main() {
  FILE *ac_file;
  // char lineToRead[5000]; 
  char *lineToRead = (char*)malloc(5000*sizeof(char)); 

  // struct node **circuit;
  // std::vector<struct node> arith_ckt;
  struct tree arith_ckt;
  // struct node n;
  int index = 0;
  
  // ac_file = fopen("examples/verysimple.ac", "r");
  ac_file = fopen("verysimple.ac", "r");
  char t = ' ';
  /*File was successfully read*/
  while (fgets(lineToRead, 5000, ac_file) != NULL) {
        
    if (*lineToRead == '(') {
      printf("\t... reading file ...\n");

      arith_ckt.nodeType = (uint64_t*)malloc(sizeof(uint64_t*) * 50000);
      arith_ckt.index = (uint64_t*)malloc(sizeof(uint64_t*) * 50000);
      arith_ckt.vr = (double*)malloc(sizeof(double*) * 50000);
      arith_ckt.dr = (double*)malloc(sizeof(double*) * 50000);
      arith_ckt.child0 = (uint64_t*)malloc(sizeof(uint64_t*) * 50000);
      arith_ckt.child1 = (uint64_t*)malloc(sizeof(uint64_t*) * 50000);
      arith_ckt.flag = (uint64_t*)malloc(sizeof(uint64_t*) * 50000);
    }
    else if (*lineToRead == 'E'){
      printf("\t... done reading file ... \n");
      index--;
      arith_ckt.dr[index] = 1;
    }
    else{
      if (*lineToRead == 'l') {
        // printf("It's a leaf node\n");
	    sscanf(lineToRead, "%s %lf", &(t), &(arith_ckt.vr[index]));
        arith_ckt.nodeType[index] = 0;
        arith_ckt.dr[index] = 0;
        arith_ckt.flag[index] = 0;
      } else if (*lineToRead == '+') {
        printf("It's a + node\n");
	  sscanf(lineToRead, "%s %ld %ld", &(t), &(arith_ckt.child0[index]), &(arith_ckt.child1[index]));
      arith_ckt.nodeType[index] = 0;
	  arith_ckt.flag[index] = 0;
	  arith_ckt.vr[index] = 0;
	  arith_ckt.dr[index] = 0;
	  
	  if (!arith_ckt.flag[arith_ckt.child0[index]]) {
	    arith_ckt.vr[index] += arith_ckt.vr[arith_ckt.child0[index]];
	  }
	  if (!arith_ckt.flag[arith_ckt.child1[index]]) {
	    arith_ckt.vr[index] += arith_ckt.vr[arith_ckt.child1[index]];
	  }
      }
      else if (*lineToRead == '*') {
      // printf("It's a * node\n");
	  sscanf(lineToRead, "%s %ld %ld", &(t), &(arith_ckt.child0[index]), &(arith_ckt.child1[index]));
      arith_ckt.nodeType[index] = 1;
      arith_ckt.vr[index] = 1;
	  arith_ckt.dr[index] = 0;

	  /*Raise bit flag if there is exactly on child with value equal to 0*/
	  if (arith_ckt.vr[arith_ckt.child0[index]] == 0 && arith_ckt.vr[arith_ckt.child1[index]] != 0) {
	    arith_ckt.flag[index] = 1; // true;
	    if (!arith_ckt.flag[arith_ckt.child1[index]]) {
	      arith_ckt.vr[index] = arith_ckt.vr[arith_ckt.child1[index]];
	    }
	    else {
	      arith_ckt.vr[index] = 0;
	    }
	  }
	  else if (arith_ckt.vr[arith_ckt.child0[index]] != 0 && arith_ckt.vr[arith_ckt.child1[index]] == 0) {
	    arith_ckt.flag[index] = 1;
	    if (!arith_ckt.flag[arith_ckt.child0[index]]) {
	      arith_ckt.vr[index] = arith_ckt.vr[arith_ckt.child0[index]];
	    }
	    else {
	      arith_ckt.vr[index] = 0;
	    }
	  }
	  else {
	    arith_ckt.flag[index] = 0; // false;
	    if (!arith_ckt.flag[arith_ckt.child0[index]]) {
	      arith_ckt.vr[index] *= arith_ckt.vr[arith_ckt.child0[index]];
	    }
	    else {
	      arith_ckt.vr[index] = 0;
	    }
	    if (!arith_ckt.flag[arith_ckt.child1[index]]) {
	      arith_ckt.vr[index] *= arith_ckt.vr[arith_ckt.child1[index]];
	    }
	    else {
	      arith_ckt.vr[index] = 0;
	    }
	  }
    }
    // arith_ckt.push_back(n);
    index++;   
    }
  }

  /*
  // i need to do this in opposite order
  for (int i = 0; i <= index; i++) {
    printf("n%d id1: %ld, id2: %ld, vr: %lf\n", i, arith_ckt.child0[i], arith_ckt.child1[i], arith_ckt.vr[i]);
  }
  printf("index: %d\n",index); 
  printf("output %lf\n\n", arith_ckt.vr[index]);
  */
  
  // backpropogation(circuit, index);
  backpropogation(arith_ckt, index);

 
  /*Free all nodes an circuit*/
  /*
  for (int i = 0; i <= index; i++) {
    //printf("n%d t: %c, dr: %lf vr: %lf\n", i, circuit[i]->nodeType, circuit[i]->dr, circuit[i]->vr);
    free(circuit[i]);
  }
    
  free(circuit);
  */
      
  /*Close file*/
  fclose(ac_file);
    
  return (EXIT_SUCCESS);
}
