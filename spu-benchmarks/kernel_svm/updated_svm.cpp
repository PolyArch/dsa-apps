#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <inttypes.h>
using namespace std;

#define VTYPE int
#define C 0.1
#define tol 0.02
#define max_passes 10
#define M 52 // number of instances
#define N 52 // number of features

std::pair<int,int> *gram_mat_pair;
int *gram_ptr;

float min(float a, float b){
    return a<b?a:b;
}
float max(float a, float b){
    return a>b?a:b;
}

// FIRST CHANGE THE DATA STRUCTURE TO SPARSE MATRIX
// Why can't y be also sparse? there is a lot of unlabled data also? let's keep it also sparse
// overwrite the data with gram matrix maybe?
void mm(std::pair<VTYPE,VTYPE> *matrix1, int *row_ptr1, std::pair<VTYPE,VTYPE> *matrix2, int *col_ptr2, int nnz1, int nnz2){
  // redundant calculation of symmetric matrix for now
  printf("gram matrix calculation call\n");
  gram_mat_pair = (std::pair<VTYPE,VTYPE>*)malloc(nnz1*nnz2*sizeof(std::pair<VTYPE,VTYPE>));
  gram_ptr = (int*)malloc((M+1)*sizeof(int)); // equal to the number of rows

  int last = -1;
  int ptr1, ptr2, end1, end2;
  int accum = 0;
  int nrows1 = M, ncols2 = M;

  for (int i=0; i<nrows1; i++){
    gram_ptr[i] = -1;
    ptr1 = row_ptr1[i];
    end1 = row_ptr1[i+1];

    for (int j=0; j<ncols2; j++){

      ptr2 = col_ptr2[j];
      end2 = col_ptr2[j+1];

      if(ptr1 == -1 || ptr2 == -1)
        continue;

      accum = 0;

      while(ptr1 <= end1 && ptr2 <= end2){
        if(matrix1[ptr1].first == matrix2[ptr2].first){
          accum += matrix1[ptr1].second*matrix2[ptr2].second;
          ptr1++; ptr2++;
        }
        else{
          if(matrix1[ptr1].first <= matrix2[ptr2].first)
            ptr1++;
          else
            ptr2++;
        }
      }

      // conditional store
      if(accum!=0){
        gram_mat_pair[++last] = std::make_pair(j,accum);
        /*if(i!=j){
            eta[i][j] += 2*accum; // do something for it: will take up a lot of space
        } else{
            eta[i][j] -= accum;
        }*/
        // col_ind3[last] = j;
        if(gram_ptr[i]==-1)
          gram_ptr[i] = last;
      }
    }
  }
  gram_ptr[M] = last;
}

// this is like both are sparse- need different definition
VTYPE dot_prod_sparse(std::pair<VTYPE,VTYPE> *data, int ptr1, int end1, int ptr2, int end2){
  VTYPE accum = 0;
  while(ptr1 <= end1 && ptr2 <= end2){
    if(data[ptr1].first == data[ptr2].first){
        accum += data[ptr1].second*data[ptr2].second;
          ptr1++; ptr2++;
        }
        else{
          if(data[ptr1].first <= data[ptr2].first)
            ptr1++;
          else
            ptr2++;
        }
      }
    return accum;
}

// this is like both are sparse
float func(float *alpha, VTYPE *y, int start_ptr, int end_ptr, int m){
  // printf("svm function call\n");
  // loop over gram_mat_pair and multiply with corresponding alpha[i]*y[i]
  float output = 0;
  /*int cur_ptr = start_ptr;
  for(int i=0; i<m && cur_ptr<end_ptr; ++i){
      if(gram_mat_pair[cur_ptr].first==i){
          output += alpha[i]*y[i]*gram_mat_pair[cur_ptr].second;
          cur_ptr++;
      }
  }*/
  int cur_index=0;
  for(int i=start_ptr; i<end_ptr; ++i){
      cur_index = gram_mat_pair[i].first;
      // printf("cur_index: %d\n",cur_index);
      // assert(cur_index<m & "index out of bounds");
      output += alpha[cur_index]*y[cur_index]*gram_mat_pair[cur_index].second;
  }
  return output;
}

VTYPE norm(std::pair<VTYPE,VTYPE> *data, int start_ptr, int end_ptr){
    VTYPE output=0;
    for(int i=start_ptr; i<end_ptr; ++i){
        output += data[i].second*data[i].second;
    }
    return output;
}

void train(std::pair<VTYPE,VTYPE> *data, int *row_ptr, int nnz, VTYPE *y){
    float alpha[M];
    float b1, b2, b=0; // initial bias?
    for(int i=0; i<M; ++i){
        alpha[i]=0.1;
    }
    float E[N]; // let's see the dim of E
    float L = 0, H = 0;
    int passes=0;
    int num_changed_alphas=0;
    float old_alpha_i=0, old_alpha_j=0;
    VTYPE eta = 0;
    float diff = 0;
    int j=0;

    mm(data, row_ptr, data, row_ptr, nnz, nnz); // how to access a symmetric sparse matrix- will be a problem?
    // gram_mat_pair, gram_ptr

    while(passes<max_passes){
        num_changed_alphas=0;
        for(int i=0; i<M; ++i){
            E[i] = func(alpha, y, gram_ptr[i], gram_ptr[i+1], M);
            // printf("Let's print the value of E[i]: %f\n",E[i]);
            if((y[i]*E[i] < -tol && alpha[i]<C) || (y[i]*E[i] > tol && alpha[i]>0)){
                // j = std::rand()%M; // should not be equal to i (random number generation in CGRA?)
                // j = (j + (j+1)%i)%M; // very complicated?
                j = (i+1)%M;
                E[j] = func(alpha, y, gram_ptr[j], gram_ptr[j+1], M);
                // printf("Let's print the value of E[j]: %f\n",E[j]);
                old_alpha_i=alpha[i];
                old_alpha_j=alpha[j];
                // eqn(10) and eqn(11)
                if(y[i] != y[j]){
                    L = max(0,alpha[j]-alpha[i]);
                    H = min(C, C+alpha[j]-alpha[i]);
                } else {
                    L = max(0, alpha[i]+alpha[j]-C);
                    H = min(C, alpha[i]+alpha[j]);
                }
                if(L==H) continue;
                // eta = eqn(14)
                // eta in dense format??- do something for it
                VTYPE inter_prod = dot_prod_sparse(data, row_ptr[i], row_ptr[i+1], row_ptr[j], row_ptr[j+1]);
                VTYPE intra_prod1 = norm(data, row_ptr[i], row_ptr[i+1]);
                VTYPE intra_prod2 = norm(data, row_ptr[j], row_ptr[j+1]);
                eta = 2*inter_prod - intra_prod1 - intra_prod2;
                // printf("Let's print the value of eta: %d\n", eta);
                // some condition also:- stream in gram matrix is better
                if(eta >= 0) continue;
                // eqn(12) and eqn(15)
                alpha[j] = alpha[j] - (y[j]*(E[i]-E[j]))/eta; // y should be stored in dense format?
                if(alpha[j]>H){
                    alpha[j]=H;
                } else if(alpha[j]<L){
                    alpha[j]=L;
                }
                diff = alpha[j]-old_alpha_j;
                if(diff < 1e-5) {
                    continue;
                }
                // eqn(16)
                alpha[i]=alpha[i]+ (y[i]*y[j])*(old_alpha_j-alpha[j]);
                // eqn(17) and eqn(18) + eqn(19)
                b1 = b - E[i] - y[i]*diff*intra_prod1 - y[j]*diff*inter_prod;
                b2 = b - E[j] - y[i]*diff*inter_prod - y[j]*diff*intra_prod2;
                if(alpha[i]>0 && alpha[i]<C){
                    b = b1;
                } else if(alpha[j]>0 && alpha[j]<C){
                    b = b2;
                } else {
                    b = (b1+b2)/2;
                }
                num_changed_alphas += 1;
            } 
        }
        if(num_changed_alphas==0){
            passes++;
        } else {
            passes=0;
        }
        printf("A pass complete\n");
    }
}

int main(){

  int nnz1=0, nrows1=0;
  int *row_ptr1;
  std::pair<VTYPE,VTYPE> *matrix1;

  FILE *m1_file;
  char lineToRead[5000];

   m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");

   int id = -1; // pointer to location in values
   int row_id = 0;
   int prev_row_id = -1;
   VTYPE t1 = 0, t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   printf("Start reading matrix1\n");
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc((nrows1+1)*sizeof(int));
           for(int i=0; i<nrows1; ++i){
               row_ptr1[i] = -1;
           }
           matrix1 = (std::pair<VTYPE,VTYPE>*)malloc(nnz1*sizeof(std::pair<VTYPE,VTYPE>));
           start = true;
       }

       else {
            ++id;
            sscanf(lineToRead, "%d %d %d", &t1, &row_id, &t2);
            row_id--;
            matrix1[id] = std::make_pair((t1-1)%N, t2%N);
            if(row_ptr1[row_id]==-1) {
                row_ptr1[row_id] = id;
                if(prev_row_id!=-1){
                  for(int i=prev_row_id+1; i<row_id; ++i){
                      // assert(row_ptr1[i]==-1 && "some problem\n");
                      row_ptr1[i] = id;
                  }
                }
                prev_row_id = row_id;
            }
       }
    }
    row_ptr1[nrows1] = nnz1-1;

    VTYPE *y;
    srand(1); // this is seed is to be used by the algorithm
    y = (VTYPE*)malloc(M*sizeof(VTYPE));
    for(int i=0; i<M; ++i){
        y[i] = rand()%10;
    }

   printf("Finished reading input data\n");

  train(matrix1, row_ptr1, nnz1, y);
  printf("svm training done\n");
  
  return 0;
}
