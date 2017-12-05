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
         double X1, X2;
         double Y1, Y2;
         double t1, t2;
         double a1, a2;
         double temp1, temp2;
         double temp;

  for (int i=0; i<N; i++) {
      if (circuit[i]->nodeType == 'n') {
	     circuit[i]->dr = 0;
	     circuit[i]->flag = false;
      }
      else if (circuit[i]->nodeType == 'v') {
	     circuit[i]->dr = 0;
	     circuit[i]->flag = false;
    }
      //code to be speedup by the softbrain
       else {
	     circuit[i]->dr = 0;
         
         a1 = circuit[i]->left->flag == 0 ? 0 : circuit[i]->left->vr;
         a2 = circuit[i]->right->flag == 0 ? 0 : circuit[i]->right->vr;
         t1 = circuit[i]->right->flag == 0 ? circuit[i]->right->vr : 0;
         t2 = circuit[i]->left->flag == 0 ? circuit[i]->left->vr : 0;

         Y1 = a1 + a2;
         //b1 = (circuit[n->child[1]]->flag) == 0 ? circuit[n->child[0]]->vr : 0;
         //b2 = (circuit[n->child[0]]->flag) == 0 ? circuit[n->child[1]]->vr : 0;

         //X2 = b1 * b2;
         X2 = t1 * t2;
         
         bool f1 = circuit[i]->left->vr == 0;
         bool f2 = circuit[i]->right->vr == 0;
         X1 = circuit[i]->left->vr == 0 ? t2 : t1;
         temp = f1 ^ f2;// (circuit[n->child[0]]->vr==0) ^ (circuit[n->child[1]]->vr==0) ;

         Y2 = temp == 0 ? X2 : X1;

         circuit[i]->flag = circuit[i]->nodeType == '+' ? false : temp;
         circuit[i]->vr = circuit[i]->nodeType == '+' ? Y1 : Y2;

    }



      //printf("node type: %c, vr: %lf, index: %d, flag %d\n", n->nodeType, n->vr, index, n->flag);
    //}
  }

  /*Print out circuit output*/
  //printf("output %lf\n\n", circuit[index]->vr);

  /*Bit-encoded backpropagation*/
  printf("\t... starting backpropagation ...\n");
  struct node* parent;
  for (int i = 0; i < N; i++) {
    parent = circuit[i];
     
       double t1, t2, Y1, Y2;
       double m1, m2, m3;
      
       t1 = (parent->vr / parent->left->vr);
       t2 = (parent->vr / parent->right->vr);
       m1 = parent->dr * parent->vr;
       
       m2 = parent->dr * t1;
       
       temp1 = parent->left->vr == 0 ? m1 : 0;
       temp2 = parent->left->vr == 0 ? 0  : m2;

       m3 = parent->dr * t2;
       X1 = parent->flag == 0 ? temp1 : m3;
       X2 = parent->flag == 0 ? temp2 : m2;

       Y1 = parent->dr == 0 ? 0 : X1;
       Y2 = parent->dr == 0 ? 0 : X2;

       parent->left->dr = parent->nodeType == '+' ? parent->dr : Y1;
       parent->right->dr = parent->nodeType == '*' ? parent->dr : Y2;
  }
  
  /*Free all nodes and circuit*/
  /*int i2;
  for (i2 = 0; i2 <= index; i2++) {
    printf("n%d t: %c, dr: %lf vr: %lf\n", i2, circuit[i2]->nodeType, circuit[i2]->dr, circuit[i2]->vr);
    free(circuit[i2]);
  }*/
    
    
}

