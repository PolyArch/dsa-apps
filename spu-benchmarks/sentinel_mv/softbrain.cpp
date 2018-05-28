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
#define INF 98239
// #define INF 100
using namespace std;

void mv_mult(std::pair<SBDT, SBDT> *matrix1, int* row_ptr1, int nrows1, int nnz1, std::pair<SBDT, SBDT> *vector2, int nnz2, int max_size) {

  std::pair<SBDT, SBDT> *vector3;
  vector3 = (std::pair<SBDT,SBDT>*)malloc(max_size*sizeof(std::pair<SBDT,SBDT>));
  for(int i=0; i<max_size; ++i) {
      vector3[i] = std::make_pair(0,0);
  }
  int temp=0;
  int last = 0;
  int ptr1 = 0, ptr2 = 0, end1 = 0, end2 = 0;

  ptr2 = 0; 
  end2 = nnz2-1;

  begin_roi();
  SB_CONFIG(merge2way_config,merge2way_size);



  SB_DMA_SCRATCH_LOAD(&vector2[ptr2], 8*2, 8*2, end2-ptr2+1, 0);
  SB_WAIT_SCR_WR();


  // let's first try for only iteration
  for (int i=0; i<190; i++){

      ptr1 = row_ptr1[i];
      end1 = i<nrows1-1 ? row_ptr1[i+1] : nnz1-1;
      // end1 = 20;
      // how to count length if end1 is -1?: may want to save length seperately
      int j=1;
      while(end1==-1) {
           end1 = i<nrows1-j ? row_ptr1[i+j] : nnz1-1;
           j++;
      };
      if(end1 == -1)
          continue;

      cout << ptr1 << " " << end1 << " " << ptr2 << " " << end2 << "\n";

      if(ptr1 == -1 || ptr2 == -1)
        continue;

      // 29-0
      SB_DMA_READ(&matrix1[ptr1], 8*2, 8*2, end1-ptr1, P_merge2way_A);
      // SB_DMA_READ(&vector2[ptr2], 8*2, 8*2, end2-ptr2+1, P_merge2way_B);
      SB_SCRATCH_READ(0, 8*2*nnz2, P_merge2way_B);
      // SB_CONST(P_merge2way_A, INF, 2);
      // SB_CONST(P_merge2way_B, INF, 2);
      SB_2D_CONST(P_merge2way_A, 98239, 1, 100, 1, 1);
      SB_2D_CONST(P_merge2way_B, 98239, 1, 100, 1, 1);

      // SB_DMA_WRITE(P_merge2way_R, 8, 8, 1, &(vector3[last].second));
      SB_DMA_WRITE(P_merge2way_R, 8, 8, 1, &temp);

      // SB_WAIT_MEM_WR(); 
      // SB_RESET();
      SB_WAIT_ALL(); 

      if(temp==0){
          cout << "accumulated sum is 0?\n";
          // --last;
      }
      else{
          cout << "i: " << i << " last: " << last << endl;
          vector3[last] = std::make_pair(i,temp);
          // vector3[last].first=i;
          last++;
      }
  }
      
  end_roi();

  cout<<"printing the output non-zero values"<<endl;
  for (int i=0; i<last; i++){
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

   // m1_file = fopen("datasets/test.mtx", "r");
   // m1_file = fopen("datasets/wy2010.mtx", "r");
   m1_file = fopen("datasets/dup_wy2010.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

   int id1 = -1; 
   int id2 = -1; // pointer to location in values
   int row_id = 0;
   uint64_t t1 = 0, t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   std::cout << "Start reading matrix1\n";
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc(nrows1*sizeof(int));
           for(int i=0; i<nrows1; ++i)
               row_ptr1[i] = -1;
           matrix1 = (std::pair<SBDT,SBDT>*)malloc(nnz1*sizeof(std::pair<SBDT,SBDT>));
           start = true;
       }

       else {
            ++id2;
            // sscanf(lineToRead, "%ld %d %ld", &matrix1[id2].first, &row_ptr1[++id1], &matrix1[id2].second);
            
            // sscanf(lineToRead, "%ld %d %ld", &matrix1[id2].first, &row_id, &matrix1[id2].second);
            sscanf(lineToRead, "%ld %d %ld", &t1, &row_id, &t2);
            row_id--;
            // matrix1[id2].first--;
            matrix1[id2] = std::make_pair(t1-1, t2);
            // row_ptr1[id1]--;
            if(row_ptr1[row_id]==-1){
                // cout << "assigning row_ptr: " << id2 << " at row_id: " << row_id << endl;
                row_ptr1[row_id] = id2;
            }
       }
    }

   std::cout << "Finished reading matrix1\n";
   // assert(id2==nnz1-1 && "number of non-zero elements in matrix does not match\n");


  // convert to CSR format take matrix2 as input
 /* id1 = -1;
  std::pair<SBDT, SBDT> *vector2;
  vector2 = (std::pair<SBDT,SBDT>*)malloc(nrows1*sizeof(std::pair<SBDT,SBDT>));

  for(int i=0; i<nrows1; ++i) {
      // if(matrix1[row_ptr1[i]].first < 100) {
      if(row_ptr1[i] != -1) {
         // if(matrix1[row_ptr1[i]].second < 1000) {
         if(matrix1[row_ptr1[i]].second < 100 || matrix1[row_ptr1[i]].second > 1000000) {
             vector2[++id1] = std::make_pair(i, matrix1[row_ptr1[i]].second);
             // cout << vector2[id1].first << " " << vector2[id1].second << "\n";
         }
      }
  }
  nnz2 = id1+1;
  for(int i=nnz2; i<nrows1; ++i)
      vector2[i] = std::make_pair(0,0);
  */


   // generating random vector
   id1 = 0;
   nnz2 = 30;
   std::pair<SBDT, SBDT> *vector2;
   vector2 = (std::pair<SBDT,SBDT>*)malloc(nnz2*sizeof(std::pair<SBDT,SBDT>));


   for(int i=0; i<nrows1; i+=nrows1/nnz2){
       if(id1==nnz2)
           break;
       vector2[id1] = std::make_pair(i, 1);
       // vector2[id1] = std::make_pair(74*i%nrows1, 3*i);
       // vector2[id1] = std::make_pair(74, 3*i);
       id1++;
    }


   // cout << id1 << " " << nnz2 << "\n";
   // assert(id1==nnz2-1 && "number of non-zero elements in vector does not match\n");
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
