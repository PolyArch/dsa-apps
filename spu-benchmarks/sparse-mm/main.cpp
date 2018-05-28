#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <assert.h>
#include <inttypes.h>
// #include "/home/vidushi/Documents/ss-stack/gem5/util/m5/m5op.h"
#include "../../common/include/sim_timing.h"

#define SBDT uint64_t
using namespace std;

/*
void begin_roi(){
    m5_dumpreset_stats(0,0);
}
void end_roi(){
    m5_dumpreset_stats(0,0);
}
*/

void mm_mult(std::pair<SBDT, SBDT> *matrix1, int* row_ptr1, int nrows1, int nnz1, std::pair<SBDT, SBDT> *matrix2, int* col_ptr2, int ncols2, int nnz2, int max_size) {

  //C in CSR format
  SBDT *col_ind3;
  SBDT *val3;
  int *row_ptr3;
  //printf("%d %d\n",nrows1,ncols2); 
  max_size = 600*600;
  val3 = (SBDT*)malloc(max_size*sizeof(SBDT));
  col_ind3 = (SBDT*)malloc(max_size*sizeof(SBDT));
  row_ptr3 = (int*)malloc(nrows1*sizeof(int));

  int last = -1;
  int ptr1, ptr2, end1, end2;
  int accum = 0;

  begin_roi();
  for (int i=0; i<nrows1; i++){
    row_ptr3[i] = -1;
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
        // insert in row i
        // cout<<"i:"<<i<<" j:"<<j<<endl;
        val3[++last] = accum;
        col_ind3[last] = j;
        if(row_ptr3[i]==-1)
          row_ptr3[i] = last;
      }


    }
  }
  end_roi();
/* 
  cout<<"printing the output non-zero values"<<endl;
  for (int i=0; i<=last; i++){
    cout<<val3[i]<<endl;
  }
*/

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
   }

   // m1_file = fopen("datasets/dup_wy2010.mtx", "r");
   // m1_file = fopen("datasets/wy2010.mtx", "r");
   m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

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
  

   std::cout << "Finished reading matrix2\n";

  int max_size = 10;
  cout<<"predicted maximum size is: "<<max_size<<endl;

  for(int i=1; i<=2; ++i) {
    mm_mult(matrix1, row_ptr1, nrows1, nnz1, matrix2, col_ptr2, ncols2, nnz2, max_size);
  }
  
  return 0;
}
