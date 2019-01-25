#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <inttypes.h>
#include <assert.h>
#include <vector>
#include <sstream>
#include <cstring>
#include <sys/time.h>

#define VTYPE uint16_t
using namespace std;


// sparse
vector<uint16_t> act_val;
vector<uint16_t> act_ind;

vector<uint16_t> wgt_val[N];
vector<uint16_t> wgt_ind[N];
uint16_t wgt_ptr[N];

uint16_t out_vec[N];

static uint64_t ticks;

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static void begin_roi() {
  ticks=rdtsc();
}

static void end_roi()   {
  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);
}

void mv_mult() {

  int ptr1, ptr2, end1, end2;
  ptr1=0; end1=0;
  ptr2 = 0; 
  end2 = act_val.size();


  VTYPE accum = 0;

  for (int i=0; i<N; i++){

    ptr1 = 0;
    end1 = wgt_val[i].size()-1;

    accum = 0;

    while(ptr1 <= end1 && ptr2 <= end2){
      if(wgt_ind[i][ptr1] == act_ind[ptr2]){
        accum += (VTYPE)(wgt_val[i][ptr1]*act_val[ptr2]);
        ptr1++; ptr2++;
      }
      else{
        if(wgt_ind[i][ptr1] <= act_ind[ptr2])
          ptr1++;
        else
          ptr2++;
      }
    }
    out_vec[i] = (VTYPE)accum;
  }
}

void read_act() {

  char lineToRead[5000];
  string str(layer_name);
  
  char r[100] = "datasets/";
  char d[100] = "/act_index.txt";
  FILE *act_ind_file2 = fopen(strcat(strcat(r,str.c_str()),d), "r");
 
  printf("Start reading act_ind activations\n");
  while(fgets(lineToRead, 5000, act_ind_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	act_ind.push_back(x);
    act_val.push_back(3);
  }  
  fclose(act_ind_file2);
 
  printf("Done reading act_ind activations\n");

}

int main(){

  char lineToRead[5000];

  string str(layer_name);

  read_act();
  printf("Done reading sparse activations\n");

  char p[100] = "datasets/";
  char b[100] = "/wgt_ptr.data";
  FILE *wgt_ptr_file2 = fopen(strcat(strcat(p,str.c_str()),b), "r");
 
  int ind=0;
  printf("Start reading wgt ptr\n");
  
  while(fgets(lineToRead, 5000, wgt_ptr_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	iss >> wgt_ptr[ind];
	ind++;
  }  
  fclose(wgt_ptr_file2);

  printf("Finished reading wgt ptr\n");
  char q[100] = "datasets/";
  char c[100] = "/wgt_val.data";
  FILE *wgt_val_file2 = fopen(strcat(strcat(q,str.c_str()),c), "r");
 
  ind=0; int k=0;
  printf("Start reading wgt val\n");
  
  printf("Start reading wgt_ind activations\n");
  char r[100] = "datasets/";
  char d[100] = "/wgt_index.data";
  FILE *wgt_ind_file2 = fopen(strcat(strcat(r,str.c_str()),d), "r");
 

  // FILE *wgt_ind_file2 = fopen(str.c_str(), "r");
  // FILE *wgt_ind_file = fopen("datasets/pyfc6_wgt_ind.txt", "r");
 
  ind=0; k=0;
  
  while(fgets(lineToRead, 5000, wgt_ind_file2) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;

	iss >> x;
	wgt_ind[k].push_back(x);
    wgt_val[k].push_back(3);
	ind++;
	// FIXME: confirm this
	if(ind==wgt_ptr[k]) {
	  k++;
	}
  }  
  fclose(wgt_ind_file2);
  

  for(int i=0; i<N; ++i) {
    out_vec[i]=0;
  }

   mv_mult();
   begin_roi();
   mv_mult();
   end_roi();

  return 0;
}

