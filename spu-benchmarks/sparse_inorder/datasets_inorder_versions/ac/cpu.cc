#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <stdbool.h>
#include <time.h>
#include <string>
#include <sstream>
#include <inttypes.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <sys/time.h>

using namespace std;

// REMEMBER TO CORRECT THIS
#define B 1
// #define thres 5
// #define thres 0

#define NUM_PART 64
#define MAX_SIZE 1000000

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





// #define NUM_THREADS 4

// const for all input (all of size V)
struct node_prop1 {
  vector<uint32_t> nodeType;
  vector<uint32_t> c0;
  vector<uint32_t> c1;
};

struct node_prop2 {
  float vr; 
  uint32_t flag;
};

int V;
vector<node_prop2> ac[B];
vector<float> ac_dr[B]; // for fixed to float point
node_prop1 ac_prop;

// start height and end height for each thread id
int start_index[NUM_PART];
int end_index[NUM_PART]; // last one is copy nodes

vector<int> height_ptr; 
vector<int> shadow_ptr; 



// parallelize across the layers (synch across each core)
void backpropagation(){
  // pid is parent id
  int c0_id, c1_id;

  for (int i = 0; i < B; ++i) {
      // for (unsigned gind=0; gind < NUM_PART; gind++) {
      for (unsigned gind=1; gind < 2; gind++) {
        int b = i - gind;
        if(b>=0){
          for(int d = start_index[gind]; d < end_index[gind]; d++) {
            for(int pid = height_ptr[d]; pid < height_ptr[d+1]; pid++) {
              c0_id = ac_prop.c0[pid];
              c1_id = ac_prop.c1[pid];

              if (ac_prop.nodeType[pid] == 0) {
                ac_dr[b][c0_id] = ac_dr[b][pid];
                ac_dr[b][c1_id] = ac_dr[b][pid];
              }
              else if (ac_prop.nodeType[pid] == 1) {
                if (ac_dr[b][pid] == 0) {
                  ac_dr[b][c0_id] = 0;
                  ac_dr[b][c1_id] = 0;
                } else if (ac[b][pid].flag) {
                  if (ac[b][c0_id].vr == 0) {
                    ac_dr[b][c0_id] = ac_dr[b][pid] * ac[b][pid].vr;
                    ac_dr[b][c1_id] = 0;
                  } else {
                    ac_dr[b][c0_id] = ac_dr[b][pid] * (ac[b][pid].vr / ac[b][c0_id].vr);
                    ac_dr[b][c1_id] = 0;
                  }
                } else {
                  ac_dr[b][c0_id] = ac_dr[b][pid] * (ac[b][pid].vr / ac[b][c0_id].vr);
                  ac_dr[b][c1_id] = ac_dr[b][pid] * (ac[b][pid].vr / ac[b][c1_id].vr);
                }
              }
            }
          }
        }
    }
  }
}



int main() {
  
  printf("Started reading file!\n");
  char lineToRead[5000];

  string str(dataset);
  char a1[100] = "datasets/";
  char b1[100] = "/final_index.data";
  FILE *hgt = fopen(strcat(strcat(a1,str.c_str()),b1), "r");
  // FILE *hgt = fopen(str.c_str(), "r");
  // FILE *hgt = fopen("datasets/final_index.data", "r");

  while(fgets(lineToRead, 5000, hgt) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
    cout << x << " ";
	height_ptr.push_back(x);
  }
  fclose(hgt); 
  cout << "DONE READING HEIGHT POINTER\n";
 
  
  // read copy nodes index
  // str = shadow_file;
  char a2[100] = "datasets/";
  char b2[100] = "/final_shadow_index.data";
  FILE *shadow = fopen(strcat(strcat(a2,str.c_str()),b2), "r");
 
  // FILE *shadow = fopen(str.c_str(), "r");
  // FILE *shadow = fopen("datasets/final_shadow_index.data", "r");

  while(fgets(lineToRead, 5000, shadow) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	int x;
	iss >> x;
    cout << x << " ";
	shadow_ptr.push_back(x);
  }
  fclose(shadow); 
  cout << "DONE READING SHADOW NODE INDICES\n";

  // CHECKME: update start and end index (using shadow ptr)

  int a=0;
  for(unsigned h=0; h<height_ptr.size();){
	// cout << a << endl;
	start_index[a/2] = h;
	h++;
	while(h<height_ptr.size() && height_ptr[h]!=shadow_ptr[a]){
	  h++;
	}
	end_index[a/2] = h;
	cout << "SE: " << start_index[a/2] << " " << end_index[a/2] << "\n";
	h++; a+=2;
  }

  cout << "DONE ASSIGNING START AND END INDEX\n";

  // read final circuit data
  // str = circuit_file;
  char a3[100] = "datasets/";
  char b3[100] = "/final_circuit.data";
  FILE *ckt = fopen(strcat(strcat(a3,str.c_str()),b3), "r");
 
  // FILE *ckt = fopen(str.c_str(), "r");
  // FILE *ckt = fopen("datasets/final_circuit.data", "r");
  int cur_v=0;

  while(fgets(lineToRead, 5000, ckt) != NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    std::string level;
    char op;
	bool valid;
	float var;
	int x,y;

	iss >> op;

	if(op=='n'){
	  iss >> level >> V;
	  for(int i=0 ; i < B; ++i){
	    ac[i].resize(V);
	    ac_dr[i].resize(V);
	  }
	  ac_prop.nodeType.resize(V);
	  ac_prop.c0.resize(V);
	  ac_prop.c1.resize(V);
	  continue;
	}

	// TODO: apply those rules here
	if(op=='l'){
	  // cout << "recognized a leaf node\n";
	  iss >> var >> valid;
	  for(int i=0 ; i < B; ++i){
		ac[i][cur_v].vr = var;
	    ac_dr[i][cur_v] = 0.0f;
	  }

	  // cout << (cur_v*NUM_PART)/V << endl;
	  ac_prop.c0[cur_v] = height_ptr[end_index[(cur_v*NUM_PART)/V]];
	  ac_prop.c1[cur_v] = height_ptr[end_index[(cur_v*NUM_PART)/V]];
	  // ac_prop.c0[cur_v] = -1;
	  // ac_prop.c1[cur_v] = -1;
	} else {
	  // iss >> ac_prop.c0[cur_v] >> ac_prop.c1[cur_v] >> valid;
	  iss >> x >> y >> valid;
	  ac_prop.c0[cur_v] = (uint32_t)x;
	  ac_prop.c1[cur_v] = (uint32_t)y;
	  for(int i=0 ; i < B; ++i){
		ac[i][cur_v].vr = 0.0f;
	    ac_dr[i][cur_v] = 0.0f;
	  }
	}
    // cout << "Child1: " << ac_prop.c0[cur_v] << " child2: " << ac_prop.c1[cur_v] << "\n"; 
	cur_v++;
  }
  fclose(ckt);  
  cout << "DONE READING CIRCUIT\n";


  printf("Done reading file!\n");

  printf("Starting backpropagation\n");
  backpropagation();
  begin_roi();
  // double start = clock();
  // backpropagation(arith_ckt, cum_nodes_at_level);
  backpropagation();
  end_roi();
  // double end = clock();
  // printf("time taken in sec is: %f\n",(end-start)/CLOCKS_PER_SEC);
  printf("Backpropagation done!\n");
  return 0;
}
