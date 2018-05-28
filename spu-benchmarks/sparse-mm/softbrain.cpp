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

void mm_mult(std::pair<SBDT, SBDT> *matrix1, int* row_ptr1, int nrows1, int nnz1, std::pair<SBDT, SBDT> *matrix2, int* col_ptr2, int ncols2, int nnz2, int max_size) {

  //C in CSR format
  SBDT *col_ind3;
  SBDT *val3;
  int *row_ptr3;
  //printf("%d %d\n",nrows1,ncols2); 
  // max_size = 600*600;
  val3 = (SBDT*)malloc(max_size*sizeof(SBDT));
  col_ind3 = (SBDT*)malloc(max_size*sizeof(SBDT));
  row_ptr3 = (int*)malloc(nrows1*sizeof(int));

  int last = 0;
  int ptr1, ptr2, end1, end2;

  begin_roi();
  SB_CONFIG(merge2way_config,merge2way_size);

  // nrows1=80;
  // ncols2=80;
  nrows1 = 11;
  // ncols2 = 5;

  for (int i=0; i<nrows1; i++){
  // for (int i=0; i<2; i++){
    row_ptr3[i] = -1; 

    SB_2D_CONST(P_merge2way_done,2,ncols2-1,0,1,1);
    ptr1 = row_ptr1[i];
    end1 = row_ptr1[i+1];

    // SB_DMA_SCRATCH_LOAD(&matrix1[ptr1], 8*2, 8*2, end1-ptr1, 0);
    // SB_WAIT_SCR_WR();

    SB_DMA_WRITE(P_merge2way_Val, 8, 8, 10000000, &val3[last]);
    SB_DMA_WRITE(P_merge2way_Index, 8, 8, 10000000, &col_ind3[last]);

    for (int j=0; j<ncols2; j++){
    // for (int j=0; j<3; j++){

      ptr2 = col_ptr2[j];
      end2 = col_ptr2[j+1];

      // cout << "i: " << i << " j: " << j << " 1: " << row_ptr1[i] << " 2: " << col_ptr2[j] << endl;
      // cout << "List 1 params: " << ptr1 << " " << end1 << "\n"; 
      // cout << "List 2 params: " << ptr2 << " " << end2 << "\n"; 

      // SB_DMA_READ(&matrix1[ptr1], 8*2, 8*2, end1-ptr1, P_merge2way_A);
      // SB_DMA_READ(&matrix2[ptr2], 8*2, 8*2, end2-ptr2+1, P_merge2way_B);
      // SB_2D_CONST(P_merge2way_A, SENTINAL, 1, 0, 1, 1);
      // SB_2D_CONST(P_merge2way_B, SENTINAL, 1, 0, 1, 1);
      SB_DMA_READ(&(matrix1[ptr1].first), 8*2, 8, end1-ptr1, P_merge2way_A);
      SB_DMA_READ(&(matrix2[ptr2].first), 8*2, 8, end2-ptr2+1, P_merge2way_B);
      SB_DMA_READ(&(matrix1[ptr1].second), 8*2, 8, end1-ptr1, P_merge2way_C);
      SB_DMA_READ(&(matrix2[ptr2].second), 8*2, 8, end2-ptr2+1, P_merge2way_D);
      SB_CONST(P_merge2way_A, SENTINAL, 1);
      SB_CONST(P_merge2way_B, SENTINAL, 1);
      SB_CONST(P_merge2way_C, 0, 1);
      SB_CONST(P_merge2way_D, 0, 1);


      SB_CONST(P_merge2way_I, j, 1);

    }

    // int nz_count = i;
    uint64_t nz_count;
    // I need option in accum to do +1(input is constant) and not discard.
    SB_RECV(P_merge2way_nz_count,nz_count);
    // wait on this output?: continue when 'only' this output is available
    SB_RESET();
    SB_WAIT_ALL(); 
    last+=nz_count;
    row_ptr3[i]=last;
    // cout << "COMPLETED COMPUTATION OF I: " << i << " with nz_count= " << nz_count << " and last as: " << last << endl;
  }

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
  int nrows1=0, ncols2=0;

  int *row_ptr1;
  std::pair<SBDT,SBDT> *matrix1;

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
   // m1_file = fopen("datasets/dup_pdb1HYS.mtx", "r");
   m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");
   // m1_file = fopen("datasets/wy2010.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

   int id1 = -1; 
   int id2 = -1; // pointer to location in values
   int row_id = 0;
   int prev_row_id = -1;
   uint64_t t1 = 0, t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   std::cout << "Start reading matrix1\n";
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc((nrows1+1)*sizeof(int));
           for(int i=0; i<nrows1; ++i){
               row_ptr1[i] = -1;
           }
           matrix1 = (std::pair<SBDT,SBDT>*)malloc(nnz1*sizeof(std::pair<SBDT,SBDT>));
           start = true;
       }

       else {
            ++id2;
            // sscanf(lineToRead, "%ld %d %ld", &matrix1[id2].first, &row_ptr1[++id1], &matrix1[id2].second);
            
            // sscanf(lineToRead, "%ld %d %ld", &matrix1[id2].first, &row_id, &matrix1[id2].second);
            sscanf(lineToRead, "%ld %d %ld", &t1, &row_id, &t2);
            row_id--;
            matrix1[id2] = std::make_pair(t1-1, t2);
            if(row_ptr1[row_id]==-1) {
                row_ptr1[row_id] = id2;
                if(prev_row_id!=-1){
                  for(int i=prev_row_id+1; i<row_id; ++i){
                      assert(row_ptr1[i]==-1 && "some problem\n");
                      row_ptr1[i] = id2;
                  }
                }
                prev_row_id = row_id;
            }
       }
    }
    row_ptr1[nrows1] = nnz1-1;


   std::cout << "Finished reading matrix1\n";




  // convert to CSR format take matrix2 as input
  int *col_ptr2;
  std::pair<SBDT, SBDT> *matrix2;
  nnz2 = nnz1;
  col_ptr2 = (int*)malloc((nrows1+1)*sizeof(int));
  matrix2 = (std::pair<SBDT,SBDT>*)malloc(nnz1*sizeof(std::pair<SBDT,SBDT>));

  ncols2 = nrows1;

  for(int i=0; i<=nrows1; ++i)
      col_ptr2[i] = row_ptr1[i];

  for(int j=0; j<nnz1; ++j)
      matrix2[j] = matrix1[j];
  

/*


  // Should have been in CSV format -- do we need conversion?

  FILE *m2_file;

  // m2_file = fopen(argv[1], "r");
  // m2_file = fopen("datasets/protein_mod.mtx", "r");
  m2_file = fopen("datasets/test.mtx", "r");

  if(!m2_file) {
      // fprintf(stderr, "Unable to read file %s\n", argv[1]);
      // return(EXIT_FAILURE);
   }
 

   std::cout << "Start reading matrix2\n";
   id1 = -1; id2 = -1; 
   start=false;
   while(fgets(lineToRead, 500000, m2_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &ncols2, &ncols2, &nnz2);
           col_ptr2 = (int*)malloc(ncols2*sizeof(int));
           matrix2 = (std::pair<SBDT,SBDT>*)malloc(nnz2*sizeof(std::pair<SBDT,SBDT>));
           start = true;
       }

       else {
            ++id2;
            sscanf(lineToRead, "%ld %d %ld", &matrix2[id2].first, &col_ptr2[++id1], &matrix2[id2].second);
            matrix2[id2].first--;
            col_ptr2[id1]--;
            if(id1!=0){
            if(row_ptr1[id1] == row_ptr1[id1-1])
                id1--;
            }
       }
    }

*/

   std::cout << "Finished reading matrix2\n";



  // can we reduce this size by some preprocessing?
  // this crosses the max_size limit
  int max_size = nrows1*ncols2;
  for (int i=0; i<nrows1; i++){
    for (int j=0; j<ncols2; j++){
      if(row_ptr1[i]==-1 || col_ptr2[j]==-1)
        max_size--;
    }
  }
  // int max_size = 10;
  cout<<"predicted maximum size is: "<<max_size<<endl;

  mm_mult(matrix1, row_ptr1, nrows1, nnz1, matrix2, col_ptr2, ncols2, nnz2, max_size);
  
  return 0;
}
