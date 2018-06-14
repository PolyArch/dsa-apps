#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <assert.h>
#include "merge2way.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

#define SBDT uint64_t
using namespace std;

void mm_mult(std::pair<SBDT, double> *matrix1, int* col_ptr1, int ncols1, std::pair<SBDT, double> *vector2, int nnz2) {

  //C in CSR format
  SBDT *col_ind3;
  double *val3;
  // int *row_ptr3;
  //printf("%d %d\n",ncols1,ncols2); 
  int max_size = 5000;
  val3 = (double*)malloc(max_size*sizeof(double));
  col_ind3 = (SBDT*)malloc(max_size*sizeof(SBDT));
  // row_ptr3 = (int*)malloc(ncols1*sizeof(int));

  // int last = 0;
  int ptr1, end1; // pt2, end1, end2;

  begin_roi();
  SB_CONFIG(merge2way_config,merge2way_size);


  // vector(1XN)*matrix(NXN) (take all the non-zero elements of the vector) =>
  // 1XN
  SB_2D_CONST(P_merge2way_done,2,ncols1-1,0,1,1);
  SB_DMA_WRITE(P_merge2way_Val, 8, 8, 10000000, &val3[0]);
  SB_DMA_WRITE(P_merge2way_Index, 8, 8, 10000000, &col_ind3[0]);
  // read vector in scr and then do this
  // SB_DMA_SCRATCH_LOAD(&(vector2[0].first), 8*2, 8, nnz2, 0);
  // SB_DMA_SCRATCH_LOAD(&(vector2[0].second), 8*2, 8, nnz2, nnz2+1);
  
  for (int i=0; i<ncols1; i++){
    ptr1 = col_ptr1[i];
    end1 = col_ptr1[i+1];
    
    SB_DMA_READ(&(matrix1[ptr1].first), 8*2, 8, end1-ptr1, P_merge2way_A);
    SB_DMA_READ(&(vector2[0].first), 8*2, 8, nnz2, P_merge2way_B);
    SB_DMA_READ(&(matrix1[ptr1].second), 8*2, 8, end1-ptr1, P_merge2way_C);
    SB_DMA_READ(&(vector2[0].second), 8*2, 8, nnz2, P_merge2way_D);

    // SB_SCRATCH_READ(0, nnz2*8, P_merge2way_B);
    // SB_SCRATCH_READ(nnz2+1, nnz2*8, P_merge2way_D);



    SB_CONST(P_merge2way_A, SENTINAL, 1);
    SB_CONST(P_merge2way_B, SENTINAL, 1);
    SB_CONST(P_merge2way_C, 0, 1);
    SB_CONST(P_merge2way_D, 0, 1);
    SB_CONST(P_merge2way_I, i, 1);
  }
  uint64_t nz_count;
  SB_RECV(P_merge2way_nz_count,nz_count);
  SB_RESET();
  SB_WAIT_ALL(); 
  end_roi();
/*
  cout<<"printing the output non-zero values"<<endl;
  for (int i=0; i<=last; i++){
    cout<<val3[i]<<endl;
  }
*/
  sb_stats();

}

int main(int argc, char** argv){

  int nnz1=0, nnz2=0;
  int ncols1=0; // , ncols2=0;

  int *col_ptr1;
  std::pair<SBDT,double> *matrix1;

  FILE *m1_file;
  char lineToRead[5000];

  /*Try to open the matrix file*/
  if(argc < 2) {
      /*No file has been passed - error*/
      // fprintf(stderr, "Must pass input matrix file\n");
      // return(EXIT_FAILURE);
   }

   // m1_file = fopen(argv[1], "r");
   // m1_file = fopen("datasets/pdb1HYS.mtx", "r");
   // m1_file = fopen("datasets/protein_mod.mtx", "r");
   // m1_file = fopen("datasets/test.mtx", "r");
   // m1_file = fopen("datasets/dup_wy2010.mtx", "r");
   m1_file = fopen("datasets/dup_pdb1HYS.mtx", "r");
   // m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");
   // m1_file = fopen("small_dataset.mtx", "r");
   // m1_file = fopen("datasets/wy2010.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

   int id2 = -1; // pointer to location in values
   int col_id = 0;
   int prev_col_id = -1;
   uint64_t t1 = 0;
   double t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   std::cout << "Start reading matrix1\n";
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &ncols1, &ncols1, &nnz1);
           col_ptr1 = (int*)malloc((ncols1+1)*sizeof(int));
           for(int i=0; i<ncols1; ++i){
               col_ptr1[i] = -1;
           }
           matrix1 = (std::pair<SBDT,double>*)malloc(nnz1*sizeof(std::pair<SBDT,double>));
           start = true;
       }

       else {
            ++id2;
            sscanf(lineToRead, "%ld %d %lf", &t1, &col_id, &t2);
            col_id--;
            matrix1[id2] = std::make_pair(t1-1, t2);
            if(col_ptr1[col_id]==-1) {
              col_ptr1[col_id] = id2;
              if(prev_col_id!=-1){
                for(int i=prev_col_id+1; i<col_id; ++i){
                    assert(col_ptr1[i]==-1 && "some problem\n");
                    col_ptr1[i] = id2;
                }
              }
              prev_col_id = col_id;
            }
       }
    }
    col_ptr1[ncols1] = nnz1-1;
    for(int j=ncols1-1; j>=0 && col_ptr1[j]==-1; j--){
      col_ptr1[j] = col_ptr1[j+1];
    }
    
    /*for(int i=0; i<ncols1; ++i) {
      printf("value of row_ptr should be in inc order: %d\n",col_ptr1[i]);
    }
    */

   std::cout << "Finished reading matrix1\n";

  // convert to CSR format take matrix2 as input
  // int *col_ptr2;
  std::pair<SBDT, double> *vector2;
  nnz2 = col_ptr1[1]; // nnz1;
  // col_ptr2 = (int*)malloc((ncols1+1)*sizeof(int));
  vector2 = (std::pair<SBDT,double>*)malloc(nnz1*sizeof(std::pair<SBDT,double>));

  // ncols2 = ncols1;

  /*
  for(int i=0; i<=ncols1; ++i){
    col_ptr2[i] = col_ptr1[i];
  }
  */

  for(int j=0; j<nnz2; ++j){
    vector2[j] = matrix1[j];
  }
  
   std::cout << "Finished reading vector2\n";


/*
  // can we reduce this size by some preprocessing?
  // this crosses the max_size limit
  int max_size = ncols1*ncols2;
  for (int i=0; i<ncols1; i++){
    for (int j=0; j<ncols2; j++){
      if(col_ptr1[i]==-1 || col_ptr2[j]==-1)
        max_size--;
    }
  }
  // int max_size = 10;
  cout<<"predicted maximum size is: "<<max_size<<endl;
  */

  mm_mult(matrix1, col_ptr1, ncols1, vector2, nnz2);
  
  return 0;
}
