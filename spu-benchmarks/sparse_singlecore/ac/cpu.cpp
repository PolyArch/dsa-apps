#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <stdbool.h>
#include <time.h>
#include <string>
#include <sstream>
#include <inttypes.h>
#include <sys/time.h>
// #include "../sim_timing.h"

#define MAX_SIZE 100000

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


struct tree {
  uint64_t* nodeType; 
  double* vr; 
  double* dr; 
  uint64_t* child0;
  uint64_t* child1;
  uint64_t* flag;
};

struct fix_tree {
  uint64_t* nodeType; 
  uint64_t* child0;
  uint64_t* child1;
};

struct input_tree {
  double* vr; 
  double* dr; 
  uint64_t* flag;
};


void forwardPropagation(fix_tree ckt, input_tree input_ckt[B], int cum_nodes_at_level[d2-d1], int levels){
  // pid is parent id
  // printf("came inside forward propagation with levels: %d\n",levels);
  int c0_id, c1_id;
  int d = levels;
  int b=0;
  // index is the leaf node here? should start from root
  
  // for (int b = 0; b < B; b++) {
    // for (int d = 0; d < (d2-d1-1); d++) {
    // for (int d = (d2-d1-1); d >= 0; d--) {
  for (d = (d-1); d >= 0; d--) {
    // printf("working at depth d: %d",d);
    // for (int d = (d-1); d >= 0; d--) {
    for(int pid = cum_nodes_at_level[d]; pid < cum_nodes_at_level[d+1]; pid++) {

      c0_id = ckt.child0[pid];
      c1_id = ckt.child1[pid];

      if (ckt.nodeType[pid] == 0) {
        input_ckt[b].flag[pid] = 0;
        input_ckt[b].vr[pid] = 0;
        input_ckt[b].dr[pid] = 0;
  
        /*Only add values if the flag is down*/
        if (!input_ckt[b].flag[c0_id]) {
          input_ckt[b].vr[pid] += input_ckt[b].vr[c0_id];
        }
        if (!input_ckt[b].flag[c1_id]) {
          input_ckt[b].vr[pid] += input_ckt[b].vr[c1_id];
        }

      }
      else if (ckt.nodeType[pid] == 1) {

        input_ckt[b].vr[pid] = 1;
        input_ckt[b].dr[pid] = 0;

        /*Raise bit flag if there is exactly one child with value equal to 0*/
        if (input_ckt[b].vr[c0_id] == 0 && input_ckt[b].vr[c1_id] != 0) {
          input_ckt[b].flag[pid] = true;
          /*Set value to product of all other non-zero child nodes*/
          if (!input_ckt[b].flag[c1_id]) {
            input_ckt[b].vr[pid] = input_ckt[b].vr[c1_id];
          }
          else {
            input_ckt[b].vr[pid] = 0;
          }
        }
        else if (input_ckt[b].vr[c0_id] != 0 && input_ckt[b].vr[c1_id] == 0) {
          input_ckt[b].flag[pid] = true;
          /*Set value to product of all other non-zero child nodes*/
          if (!input_ckt[b].flag[c0_id]) {
            input_ckt[b].vr[pid] = input_ckt[b].vr[c0_id];
          }
          else {
            input_ckt[b].vr[pid] = 0;
          }
        }
        else {
          input_ckt[b].flag[pid] = false;
          if (!input_ckt[b].flag[c0_id]) {
            input_ckt[b].vr[pid] *= input_ckt[b].vr[c0_id];
          }
          else {
            input_ckt[b].vr[pid] = 0;
          }
          if (!input_ckt[b].flag[c1_id]) {
            input_ckt[b].vr[pid] *= input_ckt[b].vr[c1_id];
          }
          else {
            input_ckt[b].vr[pid] = 0;
          }
        }
      }
    }
  }
  // printf("output of the input number %d is %lf\n", b, input_ckt[b].vr[0]);
  // }
}

void backpropagation(fix_tree ckt, input_tree input_ckt[B], int cum_nodes_at_level[d2-d1]){
  // pid is parent id
  int c0_id, c1_id;
  // index is the leaf node here? should start from root
  
  for (int b = 0; b < B; b++) {
    for (int d = 0; d < (d2-d1-1); d++) {
      // add an extra condition here that total nodes till here is
      // total_nodes/C
      for(int pid = cum_nodes_at_level[d]; pid < cum_nodes_at_level[d+1]; pid++) {
        c0_id = ckt.child0[pid];
        c1_id = ckt.child1[pid];

        if (ckt.nodeType[pid] == 0) {
          input_ckt[b].dr[c0_id] = input_ckt[b].dr[pid];
          input_ckt[b].dr[c1_id] = input_ckt[b].dr[pid];
        }
        else if (ckt.nodeType[pid] == 1) {
          if (input_ckt[b].dr[pid] == 0) {
            input_ckt[b].dr[c0_id] = 0;
            input_ckt[b].dr[c1_id] = 0;
          } else if (input_ckt[b].flag[pid]) {
            if (input_ckt[b].vr[c0_id] == 0) {
              input_ckt[b].dr[c0_id] = input_ckt[b].dr[pid] * input_ckt[b].vr[pid];
              input_ckt[b].dr[c1_id] = 0;
            } else {
              input_ckt[b].dr[c0_id] = input_ckt[b].dr[pid] * (input_ckt[b].vr[pid] / input_ckt[b].vr[c0_id]);
              input_ckt[b].dr[c1_id] = 0;
            }
          } else {
            input_ckt[b].dr[c0_id] = input_ckt[b].dr[pid] * (input_ckt[b].vr[pid] / input_ckt[b].vr[c0_id]);
            input_ckt[b].dr[c1_id] = input_ckt[b].dr[pid] * (input_ckt[b].vr[pid] / input_ckt[b].vr[c1_id]);
          }
        }
      }
    }
  }
}

int main() {
  FILE *ac_file;
  char lineToRead[MAX_SIZE]; 
  int nodes_at_level[d2-d1];
  int cum_nodes_at_level[d2-d1];
  int cur_level=0;
  struct fix_tree ckt;
  struct input_tree input_ckt[B];
  int index = 0;
  int *a, *b, *c, *d;


  ckt.nodeType = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  ckt.child0 = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  ckt.child1 = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);

  for(int i=0; i<B; ++i){
    input_ckt[i].vr = (double*)malloc(sizeof(double) * MAX_SIZE);
    input_ckt[i].dr = (double*)malloc(sizeof(double) * MAX_SIZE);
    input_ckt[i].flag = (uint64_t*)malloc(sizeof(uint64_t) * MAX_SIZE);
  }

  a = (int*)malloc(sizeof(int) * MAX_SIZE);
  b = (int*)malloc(sizeof(int) * MAX_SIZE);
  c = (int*)malloc(sizeof(int) * MAX_SIZE);
  d = (int*)malloc(sizeof(int) * MAX_SIZE);
  
  ac_file = fopen("input.data", "r");

  printf("Started reading file!\n");

  while (fgets(lineToRead, MAX_SIZE, ac_file) != NULL) {
    std::string raw(lineToRead);
    std::istringstream iss(raw.c_str());
    std::string level;
    char op;
    int n;
    iss >> level >> n >> op;
    
    nodes_at_level[cur_level] = n;
    if(cur_level==0){
      cum_nodes_at_level[cur_level] = n;
    } else {
      cum_nodes_at_level[cur_level] = (n+cum_nodes_at_level[cur_level-1]);
    }
    char ignore;
    for (int i = 0; i < n * 2; i+=2) {
        iss >> a[index] >> ignore >> b[index] >> c[index] >> ignore >> d[index];
        if(op=='*'){
          ckt.nodeType[index]=1;
        } else {
          ckt.nodeType[index]=0;
        }
        for(int i=0; i<B; ++i){
          input_ckt[i].vr[index] = rand();
          input_ckt[i].dr[index] = rand();
          input_ckt[i].flag[index] = rand()%2;
        }
        index++;
    }
    cur_level++;
  }

  int child1_ind, child2_ind;
  for(int i=0; i<index; i++){
    child1_ind = cum_nodes_at_level[a[i]-1]+b[i];
    child2_ind = cum_nodes_at_level[c[i]-1]+d[i];

    ckt.child0[i] = child1_ind;
    ckt.child1[i] = child2_ind;

  }
 
  // int no_nodes_per_cgra = 1484;
  // int no_nodes_per_core = index/(cores_per_spad*num_pe);
  // int no_nodes_per_core = index/cores_per_spad;
  // int no_nodes_per_core = 2192/cores_per_spad+1;
  // int no_nodes_per_core = 3340/cores_per_spad+1;
  int no_nodes_per_core = 1484/cores_per_spad+1;
  
  printf("Done reading file!\n");

  int levels=d2-d1-1;
  // turn it off just to check forward propagation quickly
  for(int i=0; i<(d2-d1-1); i++){
    if(cum_nodes_at_level[i]>no_nodes_per_core){
      cum_nodes_at_level[i]=no_nodes_per_core;
      levels=i+1;
      break;
    }
  }
  // for(int i=(d2-d1-2); i>=0; i--){
  //   if(cum_nodes_at_level[i]>no_nodes_per_core){
  //     cum_nodes_at_level[i]=no_nodes_per_core;
  //     levels=i+1;
  //     break;
  //   }
  // }
 
  // printf("Done reading file!\n");

  forwardPropagation(ckt, input_ckt, cum_nodes_at_level, levels);
  printf("Starting forward propagation with number of nodes: %d\n", index);
  printf("num of levels: %d effective nodes: %d\n", levels, no_nodes_per_core);
  begin_roi();
  forwardPropagation(ckt, input_ckt, cum_nodes_at_level, levels);
  end_roi();
  printf("Forward propagation done!\n");
 

  // printf("Starting backpropagation with number of nodes: %d\n", index);
  // backpropagation(ckt, input_ckt, cum_nodes_at_level);
  // printf("Backpropagation done!\n");
  return 0;
}
