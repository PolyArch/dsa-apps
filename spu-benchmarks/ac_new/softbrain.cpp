#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/* Reads an .ac file by argument and calculates the circuit output and 
 * partial derivatives for every node.
 * The circuit is stored in an adjacency list.
 * Each non-leaf node storess its children in a separate array.
 */

/* Node Structure
   Children and flag only apply to non-leaf nodes*/
struct node {
  /*Node type can be 'n' or 'v' for leaf nodes
    Node type can be '+' or '-' for non-leaf nodes */
  char nodeType; 
  /*Node index, e.g. "Third variable: n=2*/
  int index;
  /*Value of the node*/
  double vr; 
  /*Derivative of the node*/
  double dr; 
  /*Children nodes*/
  /*Currently assumes a binary AC (only two children per node)*/
  //struct node **child;
  int child[2];
  /*Bit flag, true means there is exactly one child that is zero*/
  bool flag;
  /*Zero counter, if this counter is 1, set flag to true*/
  /*Currently not used (assuming binary AC structure)*/
  //int counter;
};

void backpropogation(struct node**circuit, int index){
  /*Bit-encoded backpropagation*/
  printf("\t... starting backpropagation ...\n");

  SB_CONFIG(test_config,test_size);

  SB_DMA_READ(&circuit[0]->nodeType,8,8*sizeof(struct node*),index+1,P_test_A);
  SB_DMA_READ(&circuit[0]->dr,8,8*sizeof(struct node*),index+1,P_test_dr);
  SB_DMA_READ(&circuit[0]->flag,8,8*sizeof(struct node*),index+1,P_test_flag);
  SB_DMA_READ(&circuit[0]->vr,8,8*sizeof(struct node*),index+1,P_test_vr);
  SB_DMA_READ(&circuit[0]->child[0],8,8*sizeof(struct node*),index+1,P_IND_1);
  SB_DMA_READ(&circuit[0]->child[1],8,8*sizeof(struct node*),index+1,P_IND_2);
  SB_CONST(P_test_const,0,index+1);
  SB_INDIRECT_CONFIG(T64,T64,1);
  SB_INDIRECT1(P_IND_1, &circuit[0]->vr, index+1, P_test_c1vr);
  SB_INDIRECT1(P_IND_2, &circuit[0]->vr, index+1, P_test_c2vr);

  SB_INDIRECT_WR(P_IND_1, &circuit[0]->dr, index+1, P_test_c0dr));
  SB_INDIRECT_WR(P_IND_2, &circuit[1]->dr, index+1, P_test_c1dr));

  SB_WAIT_ALL();
  sb_stats();
  
}

int main(int argc, char** argv) {
  FILE *ac_file;
  char lineToRead[50000]; 
  struct node **circuit;
  struct node *n;
  int index = 0;
  int temp = 0;
  
  /*Try to open the AC file*/
  if (argc < 2) {
    /*No file has been passed - error*/
    fprintf(stderr, "Must pass AC file\n");
    return(EXIT_FAILURE);
  }
    
  ac_file = fopen(argv[1], "r");
    
  if (!ac_file) {
    /* File does not exist*/
    fprintf(stderr, "Unable to read file %s\n", argv[1]);
    return(EXIT_FAILURE);
  }
    
  /*File was successfully read*/
  while (fgets(lineToRead, 50000, ac_file) != NULL) {
    //printf("%s", lineToRead);
        
    if (*lineToRead == '(') {
      printf("\t... reading file ...\n");
      /*Allocate memory for the circuit*/
      circuit = (struct node**)malloc(sizeof(struct node*) * 50000);
    }
    else if (*lineToRead == 'E'){
      printf("\t... done reading file ... \n");
      index--;
      n->dr = 1;
    }
    else{
      if (*lineToRead == 'l') {
	/*Leaf node (Constant)*/
	/*Insert node into circuit*/
	n = (struct node*)malloc(sizeof(struct node));
	sscanf(lineToRead, "%s %lf", &(n->nodeType), &(n->vr));
	n->dr = 0;
	n->flag = false;
      }
/*
      else if (*lineToRead == 'v') {
	n = (struct node*)malloc(sizeof(struct node));
	sscanf(lineToRead, "%s %d %lf", &(n->nodeType), &(n->index), &(n->vr));
	n->dr = 0;
	n->flag = false;
      }*/
      else if (*lineToRead == '+') {
	/*Non-leaf (Operation)*/
	n = (struct node*)malloc(sizeof(struct node));
	/*"n->child" stores the index of the children nodes in the circuit*/
	sscanf(lineToRead, "%d %s %d %d", &temp, &(n->nodeType), &(n->child[0]), &(n->child[1]));
	n->flag = false;
	n->vr = 0;
	n->dr = 0;
	
	/*Only add values if the flag is down*/
	if (!circuit[n->child[0]]->flag) {
	  n->vr += circuit[n->child[0]]->vr;
	}
	if (!circuit[n->child[1]]->flag) {
	  n->vr += circuit[n->child[1]]->vr;
	}
	/*Incorrect output when using bit flags*/
	//n->vr = circuit[n->child[0]]->vr + circuit[n->child[1]]->vr;
      }
      else if (*lineToRead == '*') {
	/*Non-leaf (Operation)*/
	n = (struct node*)malloc(sizeof(struct node));
	sscanf(lineToRead, "%d %s %d %d", &temp, &(n->nodeType), &(n->child[0]), &(n->child[1]));
        n->vr = 1;
	n->dr = 0;

	/*Raise bit flag if there is exactly one child with value equal to 0*/
	if (circuit[n->child[0]]->vr == 0 && circuit[n->child[1]]->vr != 0) {
	  n->flag = true;
	  /*Set value to product of all other non-zero child nodes*/
	  if (!circuit[n->child[1]]->flag) {
	    n->vr = circuit[n->child[1]]->vr;
	  }
	  else {
	    n->vr = 0;
	  }
	}
	else if (circuit[n->child[0]]->vr != 0 && circuit[n->child[1]]->vr == 0) {
	  n->flag = true;
	  /*Set value to product of all other non-zero child nodes*/
	  if (!circuit[n->child[0]]->flag) {
	    n->vr = circuit[n->child[0]]->vr;
	  }
	  else {
	    n->vr = 0;
	  }
	}
	else {
	  n->flag = false;
	  if (!circuit[n->child[0]]->flag) {
	    n->vr *= circuit[n->child[0]]->vr;
	  }
	  else {
	    n->vr = 0;
	  }
	  if (!circuit[n->child[1]]->flag) {
	    n->vr *= circuit[n->child[1]]->vr;
	  }
	  else {
	    n->vr = 0;
	  }
	}
      }
      //printf("node type: %c, vr: %lf, index: %d, flag %d\n", n->nodeType, n->vr, index, n->flag);
      circuit[index] = n;
      index++;   
    }
  }

  /*Print out circuit output*/
  printf("output %lf\n\n", circuit[index]->vr);

 
  /*Free all nodes and circuit*/
  for (int i = 0; i <= index; i++) {
    //printf("n%d t: %c, dr: %lf vr: %lf\n", i, circuit[i]->nodeType, circuit[i]->dr, circuit[i]->vr);
    free(circuit[i]);
  }
    
  free(circuit);
      
  /*Close file*/
  fclose(ac_file);
    
  return (EXIT_SUCCESS);
}
  
