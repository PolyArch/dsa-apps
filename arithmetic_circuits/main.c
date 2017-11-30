#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#define MAX_Q_SIZE 500
#include "bottom_up.h"
#include "top_down.h"
#include "../common/include/sb_insts.h"
#include "../common/include/sim_timing.h"
#define H 1000
#define N 1000

struct node {
  bool nodeType;
  int index;
  double vr;
  double dr;
  bool flag;
  int left;
  int right;
};

int indexh[H]; //to store nuber of nodes at each height
struct node *circuit; //assuming max height of the arithmetic circuit
int index = 0;
uint64_t sort_nodes[H][N]; //assuming max N nodes at a height
uint64_t sort_left[H][N];
uint64_t sort_right[H][N];

int max(int x, int y){
  return x>y?x:y;
}

void arith_ckt(int h, int *indexH, int* n_right) {
  printf("...bottom up propogation begins with circuit of height...%d\n",h);
  //printf("%llu\n",&circuit[0].vr); 
  //printf("%llu\n",&circuit[1].vr); 
  //printf("%d\n",sort_left[2][0]);
  //printf("%d\n",sort_right[2][0]);
  //printf("%d\n",sort_nodes[2][0]);
  begin_roi();
  
  SB_CONFIG(bottom_up_config, bottom_up_size);

  for (int i=1; i<=h; i++) {
//   printf("%d %d %d %d\n",i,indexH[i],sizeof(&circuit[0].nodeType+sort_nodes[1][5]),sizeof(&circuit[index].nodeType));

    SB_STRIDE(8,8);//access size has to be 8: instead of &a[0]+4=&a[4], &a[0]+4*sizeof(node)
    SB_DMA_READ_SIMP(&sort_left[i][0], indexH[i], P_IND_1);
    SB_DMA_READ_SIMP(&sort_right[i][0], indexH[i], P_IND_2);
    SB_DMA_READ_SIMP(&sort_nodes[i][0], indexH[i], P_IND_3);
    SB_DMA_READ_SIMP(&sort_left[i][0], indexH[i], P_IND_4);
    SB_DMA_READ_SIMP(&sort_right[i][0], indexH[i], P_IND_TRIP0);

    SB_INDIRECT64(P_IND_1,&(circuit[0].vr),indexH[i],P_bottom_up_c0_vr);  
    SB_INDIRECT64(P_IND_2,&(circuit[0].vr),indexH[i],P_bottom_up_c1_vr);  

    //SB_STRIDE(8,1);//access size has to be 8, 3:8 bit
    SB_INDIRECT64(P_IND_3,&(circuit[0].nodeType),indexH[i],P_bottom_up_node);  
    SB_INDIRECT64(P_IND_4,&(circuit[0].flag),indexH[i],P_bottom_up_c0_flag);  
    SB_INDIRECT64(P_IND_TRIP0,&(circuit[0].flag),indexH[i],P_bottom_up_c1_flag);  
    SB_CONST(P_bottom_up_const, 0, indexH[i]);
    //#define SB_INDIRECT_WR(ind_port, addr_offset, type, num_elem, output_port)
    //SB_DMA_WRITE_SIMP(output_port, num_strides, mem_addr)

    //for temporary results: comment these 2 lines
    SB_DMA_WRITE(P_bottom_up_n_vr, 8, 8, indexH[i], &circuit[0].vr); 
    SB_DMA_WRITE(P_bottom_up_n_flag, 8, 8, indexH[i], &circuit[0].flag); 
  
    //correct execution: TODO 
    /*SB_DMA_READ_SIMP(&sort_nodes[i][0], indexH[i], P_IND_DOUB0);
    SB_DMA_READ_SIMP(&sort_nodes[i][0], indexH[i], P_IND_DOUB1);
    SB_INDIRECT64_WR(P_IND_DOUB0, &circuit[0].vr, indexH[i], P_bottom_up_n_vr);
    SB_INDIRECT64_WR(P_IND_DOUB1, &circuit[0].flag, indexH[i], P_bottom_up_n_flag);
   */     


    /*for (int j=0; j <indexH[j]; j++) {
    //SB_DMA_READ(&(circuit[0].vr)+sort_nodes[i][j],sizeof(struct node), 1, 1 , P_bottom_up_node); 
    SB_DMA_WRITE(P_bottom_up_n_vr, sizeof(struct node), 8, 1, &(circuit[0].vr)+sort_nodes[i][j]); 
    SB_DMA_WRITE(P_bottom_up_n_flag, sizeof(struct node), 1, 1, &(circuit[0].flag)+sort_nodes[i][j]); 
   }
*/
    SB_WAIT_ALL();

  }

end_roi();
/*  printf("...top down propogation begins...\n");
  //begin_roi();
  SB_CONFIG(top_down_config, top_down_size);

  for(int i=h; i>0; i--) {
    SB_STRIDE(8,8);//access size has to be 8: instead of &a[0]+4=&a[4], &a[0]+4*sizeof(node)
    SB_DMA_READ_SIMP(&sort_left[i][0],indexH[i],P_IND_1);
    SB_INDIRECT64(P_IND_1,&(circuit[0].vr),indexH[i],P_top_down_c0_vr);  

    SB_DMA_READ_SIMP(&sort_right[i][0],indexH[i],P_IND_2);
    SB_INDIRECT64(P_IND_2,&(circuit[0].vr),indexH[i],P_top_down_c1_vr);  

    SB_DMA_READ_SIMP(&sort_nodes[i][0],indexH[i],P_IND_3);
    SB_INDIRECT64(P_IND_3,&(circuit[0].vr),indexH[i],P_top_down_p_vr);  
    SB_DMA_READ_SIMP(&sort_nodes[i][0],indexH[i],P_IND_4);
    SB_INDIRECT64(P_IND_4,&(circuit[0].dr),indexH[i],P_top_down_p_dr);  
    SB_DMA_READ_SIMP(&sort_nodes[i][0],indexH[i],P_IND_TRIP0);
    SB_INDIRECT64(P_IND_TRIP0,&(circuit[0].nodeType),indexH[i],P_top_down_node);  
    SB_DMA_READ_SIMP(&sort_nodes[i][0],indexH[i],P_IND_TRIP1);
    SB_INDIRECT64(P_IND_TRIP1,&(circuit[0].flag),indexH[i],P_top_down_p_flag);  


    SB_CONST(P_top_down_const, 0, indexH[i]);
    
    //indirect write: TODO
    SB_DMA_WRITE(P_top_down_c0_dr, sizeof(struct node), 8, indexH[i], &(circuit[0].dr)); 
    SB_DMA_WRITE(P_top_down_c1_dr, sizeof(struct node), 8, indexH[i], &(circuit[0].dr)); 

    SB_WAIT_ALL();
  }
*/

}



int height_balance(FILE* ac_file) {
  char lineToRead[5000];
  struct node n; 
  //int index = 0;
  int x=0;
  int height[H];

  for (int i=0; i<H; i++) {
      indexh[i] = 0;
  }
   for (int i=0; i<H; i++) {
     for(int j=0; j<H; j++){
      sort_nodes[i][j] = 0;
      sort_left[i][j] = 0;
      sort_right[i][j] = 0;
  }}
  char a;
  while (fgets(lineToRead, 5000, ac_file) != NULL) {
        
    if (*lineToRead == '(') {
      printf("\t... reading file ...\n");
      circuit = (struct node*)malloc(5000);
    }

    else if (*lineToRead == 'E'){
      printf("\t... done reading file ... \n");
      index--;
    }

    else{
      //printf("came here\n");
      //n = (struct node)malloc(sizeof(struct node));
      if (*lineToRead == 'n') {
	sscanf(lineToRead, "%s %lf", &(a), &(n.vr));
        height[index] = 0;
        n.nodeType = 0;
        n.index = 0;
        n.flag = 0;
        n.left = 0; n.right = 0;
        n.vr = 0; n.dr = 0;
      }
      else if (*lineToRead == 'v') {
	sscanf(lineToRead, "%s %d %lf", &(a), &(n.index), &(n.vr));
        height[index] = 0;
        n.nodeType = 0;
        n.index = 0;
        n.flag = 0;
        n.left = 0; n.right = 0;
        n.vr = 0; n.dr = 0;
      }
      else if (*lineToRead == '+') {
	sscanf(lineToRead, "%s %d %d", &(a), &n.left, &n.right);
        height[index] = max(height[n.left], height[n.right])+1;
        n.nodeType = 0;
        n.index = 0;
        n.flag = 0;
        n.vr = 0; n.dr = 0;
      }
      else if (*lineToRead == '*') {
	sscanf(lineToRead, "%s %d %d", &(a), &n.left, &n.right);
        height[index] = max(height[n.left], height[n.right])+1;
        n.nodeType = 1;
        n.index = 0;
        n.flag = 0;
        n.vr = 0; n.dr = 0;
      }
      circuit[index] = n; 
      sort_nodes[height[index]][indexh[height[index]]] = (index)*sizeof(struct node);
      sort_left[height[index]][indexh[height[index]]] = n.left*sizeof(struct node);
      sort_right[height[index]][indexh[height[index]]] = n.right*sizeof(struct node);
      //printf("%d  %d\n",height[index],sort_nodes[height[index]][indexh[height[index]]-1]);
      //printf("%d %d %d %d   %d\n",index,n.left,n.right,height[index], indexh[height[index]]);
     
      indexh[height[index]]++;
      index++;   

     }
 }
  /*Close file*/
  fclose(ac_file);

  //printf("check here: %d\n", height[index]);
  return height[index];    
  }




int main(int argc, char** argv) {
  FILE *ac_file;
  
  /*if (argc < 2) {
    fprintf(stderr, "Must pass AC file\n");
    return(EXIT_FAILURE);
  }*/

  //ac_file = fopen(argv[1], "r");
  //ac_file = fopen("examples/verysimple.ac", "r");
  ac_file = fopen("examples/example.ac", "r");
  if (!ac_file) {
    fprintf(stderr, "Unable to read file %s\n", argv[1]);
    return(EXIT_FAILURE);
  }
  
  int h = height_balance(ac_file);

  int n_right[h+1];
  n_right[h] = indexh[h];
  for (int i=h-1;i>=0;i--){
    n_right[i] = n_right[i+1] + indexh[i];
    //printf("%d\n",index-n_right[i]);
  }
 
  //double begin = clock();
  arith_ckt(h, indexh, n_right);
  //double end = clock();
  //double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  //printf("time in calculation of values: %lf\n", time_spent);

  return 0;
}
