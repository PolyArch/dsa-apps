#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

struct node {
  char nodeType;
  int index; //required only for 'v' node
  double vr;
  double dr;
  bool flag;
  struct node *left;
  struct node *right;
};

void arith_ckt(struct node** circuit, int H, int *indexH, int N,double *a[],double *b[],double *c[],double *d[]) {
   
      for (int i=0; i<N; i++) {
      if (circuit[i]->nodeType == 'n') {
	circuit[i]->dr = 0;
	circuit[i]->flag = false;
      }
      else if (circuit[i]->nodeType == 'v') {
	circuit[i]->dr = 0;
	circuit[i]->flag = false;
      }
      else if (circuit[i]->nodeType == '+') {
	circuit[i]->flag = false;
	circuit[i]->vr = 0;
	circuit[i]->dr = 0;
	
	if (!circuit[i]->left->flag) {
	  circuit[i]->vr += (circuit[i]->left)->vr;
	}
	if (!circuit[i]->right->flag) {
	  circuit[i]->vr += (circuit[i]->right)->vr;
	}
	/*Incorrect output when using bit flags*/
	//n->vr = circuit[n->child[0]]->vr + circuit[n->child[1]]->vr;
      }
      else if (circuit[i]->nodeType == '*') {
        circuit[i]->vr = 1;
	circuit[i]->dr = 0;

	/*Raise bit flag if there is exactly one child with value equal to 0*/
	if ((circuit[i]->left)->vr == 0 && (circuit[i]->right)->vr != 0) {
	  circuit[i]->flag = true; 
	  /*Set value to product of all other non-zero child nodes*/
	  if (!circuit[i]->right->flag) {
	    circuit[i]->vr = circuit[i]->right->vr;
	  }
	  else {
	    circuit[i]->vr = 0;
	  }
	}

	else if (circuit[i]->left->vr != 0 && circuit[i]->right->vr == 0) {
	  circuit[i]->flag = true;
	  /*Set value to product of all other non-zero child nodes*/
	  if (!circuit[i]->left->flag) {
	    circuit[i]->vr = circuit[i]->left->vr;
	  }
	  else {
	    circuit[i]->vr = 0;
	  }
	}
	else {
	  circuit[i]->flag = false;
	  if (!circuit[i]->left->flag) {
	    circuit[i]->vr *= circuit[i]->left->vr;
	  }
	  else {
	    circuit[i]->vr = 0;
	  }
	  if (!circuit[i]->right->flag) {
	    circuit[i]->vr *= circuit[i]->right->vr;
	  }
	  else {
	    circuit[i]->vr = 0;
	  }
	}
      }
      printf("node type: %c, vr: %lf, flag %d\n", circuit[i]->nodeType, circuit[i]->vr, circuit[i]->flag);
    }
//  }

  /*Print out circuit output*/
  //printf("output %lf\n\n", circuit[index]->vr);

  /*Bit-encoded backpropagation*/
  printf("\t... starting backpropagation ...\n");
  struct node* parent;
  for (int i = 0; i < N; i++) {
    parent = circuit[i];

    /*Assign dr values depending on parent node*/
    if (parent->nodeType == '+') {
      parent->left->dr = parent->dr;
      parent->right->dr = parent->dr;
    }
    else if (parent->nodeType == '*') {
      /*if bit flag is down, and parent is non-zero, dr(c) = dr(p)*vr(p)/vr(c)*/
      if (parent->dr == 0) {
	/*Set all child nodes dr to zero*/
	parent->left->dr = 0;
	parent->right->dr = 0;
      }
      else if (parent->flag) {
	/*Check value of all child nodes*/
	/*if flag is up and child is zero, then dr(c) = dr(p) * vr(p)*/
	if (parent->left->vr == 0) {
	  parent->left->dr = parent->dr * parent->vr;
	  /*Set all other children dr to zero*/
	  parent->right->dr = 0;
	}
	else {
	  parent->right->dr = 0;
	  parent->left->dr = parent->dr *
	    (parent->vr / parent->left->vr);
	}
      }
      else {
	parent->right->dr = parent->dr *
	  (parent->vr / parent->right->vr);
	parent->left->dr = parent->dr *
	  (parent->vr / parent->left->vr);
      }
    }
  }
  printf("completed backpropogation\n");
  
    
}
  

