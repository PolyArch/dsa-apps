// #include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include "merge2way.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

#define SBDT uint64_t
// using namespace std;

void mv_mult(std::pair<SBDT, SBDT> *matrix1, int* row_ptr1, int nrows1, int nnz1, std::pair<SBDT, SBDT> *vector2, int nnz2, int max_size) {

  std::pair<SBDT, SBDT> *vector3;
  vector3 = (std::pair<SBDT,SBDT>*)malloc(max_size*sizeof(std::pair<SBDT,SBDT>));
  for(int i=0; i<max_size; ++i) {
      vector3[i] = std::make_pair(-1,-1);
  }
  int last = -1;
  int ptr1, ptr2, end1, end2;
  int accum = 0;

  // begin_roi();

  for (int i=0; i<nrows1; i++){

      ptr1 = row_ptr1[i];
      end1 = i<nrows1-1 ? row_ptr1[i+1] : nnz1-1;
      ptr2 = 0; 
      end2 = nnz2-1;

      if(ptr1 == -1 || ptr2 == -1)
        continue;

      accum = 0;

      while(ptr1 <= end1 && ptr2 <= end2){
        if(matrix1[ptr1].first == vector2[ptr2].first){
          accum += matrix1[ptr1].second*vector2[ptr2].second;
          ptr1++; ptr2++;
        }
        else{
          if(matrix1[ptr1].first <= vector2[ptr2].first)
            ptr1++;
          else
            ptr2++;
        }
      }

      if(accum!=0){
        // insert in row i
        vector3[++last] = std::make_pair(i, accum);
      }

  }
      
  // end_roi();

  printf("printing the output non-zero values\n");
  for (int i=0; i<=last; i++){
      printf("%ld ",vector3[i].second);
  }

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
   m1_file = fopen("datasets/wy2010.mtx", "r");

   if(!m1_file) {
       /* File does not exist */
       fprintf(stderr, "Unable to read file %s\n", argv[1]);
       return(EXIT_FAILURE);
    }
    

   int id1 = -1; int id2 = -1; 
   bool start=false;
   // int x = 0, y = 0;
   // std::cout << "Start reading matrix1\n";
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

   // std::cout << "Finished reading matrix1\n";


  // convert to CSR format take matrix2 as input
  id1 = -1;
  std::pair<SBDT, SBDT> *vector2;
  vector2 = (std::pair<SBDT,SBDT>*)malloc(nrows1*sizeof(std::pair<SBDT,SBDT>));

  for(int i=0; i<nrows1; ++i) {
      // if(matrix1[row_ptr1[i]].first < 40) {
      if(matrix1[row_ptr1[i]].second < 1000) {
          vector2[++id1] = std::make_pair(i, matrix1[row_ptr1[i]].second);
      }
  }
  nnz2 = id1;
  
  printf("size of the vector is: %d\n",nnz2);
   // std::cout << "Finished getting vector1\n";

  // can we reduce this size by some preprocessing?
  int max_size = nrows1;
  for (int i=0; i<nrows1; i++){
      if(row_ptr1[i]==-1)
        max_size--;
    }
  printf("predicted maximum size is: %d\n",max_size);

  mv_mult(matrix1, row_ptr1, nrows1, nnz1, vector2, nnz2, max_size);
  
  return 0;
}
