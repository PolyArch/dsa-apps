#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include "merge2way.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

#define SBDT uint64_t
using namespace std;

void mv_mult(std::pair<SBDT, SBDT> *matrix1, int* row_ptr1, int nrows1, int nnz1, std::pair<SBDT, SBDT> *vector2, int nnz2, int max_size) {

  std::pair<SBDT, SBDT> *vector3;
  vector3 = (std::pair<SBDT,SBDT>*)malloc(max_size*sizeof(std::pair<SBDT,SBDT>));
  for(int i=0; i<max_size; ++i) {
      vector3[i] = std::make_pair(-1,-1);
  }
  int last = -1;
  /*
  int ptr1, ptr2, end1, end2;
  ptr2 = 0; 
  end2 = nnz2-1;
  */
  int ptr1, end1;

  begin_roi();
  SB_CONFIG(merge2way_config,merge2way_size);
  
  // read vector into matrix first
  SB_DMA_SCRATCH_LOAD(&vector2[0], 8*2, 8*2, nnz2, 0);
  SB_WAIT_ALL();

  for (int i=0; i<nrows1; i++){

      ptr1 = row_ptr1[i];
      end1 = i<nrows1-1 ? row_ptr1[i+1] : nnz1-1;
      
      if(ptr1 == -1)
        continue;

      SB_DMA_READ(&matrix1[ptr1], 8*2, 8*2, end1-ptr1+1, P_merge2way_A);
      // SB_SCR_PORT_STREAM(0, 8*2, 8*2, end2-ptr2+1, P_merge2way_B);
      SB_SCR_PORT_STREAM(0, 8*2, 8*2, nnz2, P_merge2way_B);
      // SB_DMA_READ(&vector2[ptr2], 8*2, 8*2, end2-ptr2+1, P_merge2way_B);
      SB_CONST(P_merge2way_A, 100, 2);
      SB_CONST(P_merge2way_B, 100, 2);

      SB_DMA_WRITE(P_merge2way_R, 8, 8, 1, &vector3[++last].second);

      SB_WAIT_MEM_WR(); 
      SB_RESET();
      SB_WAIT_ALL(); 

      if(vector3[last].second==0){
          --last;
      }
      else{
          // cout << "i: " << i << " j: " << j << endl;
          vector3[last].first=i;
      }
  }
      
  end_roi();

  cout<<"printing the output non-zero values"<<endl;
  for (int i=0; i<=last; i++){
    cout<<vector3[i].second<<endl;
  }

  sb_stats();

}

int main(int argc, char** argv){

  int nnz1=0, nnz2=0;
  int nrows1=0; //, ncols2=0;

  int *row_ptr1;
  std::pair<SBDT,SBDT> *matrix1;

  FILE *m1_file;
  char lineToRead[5000];

  if(argc < 2) {
   }

   m1_file = fopen("datasets/test.mtx", "r");
   // m1_file = fopen("datasets/wy2010.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

   int id1 = -1; int id2 = -1; 
   bool start=false;
   // int x = 0, y = 0;
   std::cout << "Start reading matrix1\n";
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc(nrows1*sizeof(int));
           matrix1 = (std::pair<SBDT,SBDT>*)malloc(nnz1*sizeof(std::pair<SBDT,SBDT>));
           start = true;
       }

       else {
            ++id2;
            sscanf(lineToRead, "%ld %d %ld", &matrix1[id2].first, &row_ptr1[++id1], &matrix1[id2].second);
            matrix1[id2].first--;
            row_ptr1[id1]--;
            if(id1!=0){
            if(row_ptr1[id1] == row_ptr1[id1-1])
                id1--;
            }
       }
    }

   std::cout << "Finished reading matrix1\n";

/*
  // convert to CSR format take matrix2 as input
  id1 = -1;
  std::pair<SBDT, SBDT> *vector2;
  vector2 = (std::pair<SBDT,SBDT>*)malloc(nrows1*sizeof(std::pair<SBDT,SBDT>));

  for(int i=0; i<nrows1; ++i) {
      if(matrix1[row_ptr1[i]].first == 0) {
          vector2[++id1] = std::make_pair(i, matrix1[row_ptr1[i]].second);
      }
  }
  nnz2 = id1;
*/  
// generating random vector
   id1 = -1;
   nnz2 = 30;
   std::pair<SBDT, SBDT> *vector2;
   vector2 = (std::pair<SBDT,SBDT>*)malloc(nnz2*sizeof(std::pair<SBDT,SBDT>));


   for(int i=0; i<nrows1; i+=nrows1/nnz2){
       vector2[++id1] = std::make_pair(i, 1000*i);
    }



   std::cout << "Finished getting vector1\n";

  // can we reduce this size by some preprocessing?
  int max_size = nrows1;
  for (int i=0; i<nrows1; i++){
      if(row_ptr1[i]==-1)
        max_size--;
    }
  cout<<"predicted maximum size is: "<<max_size<<endl;

  mv_mult(matrix1, row_ptr1, nrows1, nnz1, vector2, nnz2, max_size);
  
  return 0;
}
