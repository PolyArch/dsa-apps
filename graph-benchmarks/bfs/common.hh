#ifndef _COMMON_H
#define _COMMON_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#include <bitset>
#include <string.h>
#include <list>
#include <algorithm>
#include "assert.h"
using namespace std;

#define INF 100000
#define MAX_ITER 100000000
// #define MAX_ITER 50
// #define SCRATCH_SIZE 1048576
#define SCRATCH_SIZE 4048576
// #define SCRATCH_SIZE 16384
// #define SCRATCH_SIZE 4096
#define LOCAL_SCRATCH_SIZE (SCRATCH_SIZE/core_cnt)
// can change this with the model to save memory
#define MAX_TIMESTAMP 10000000 // window size
// #define MAX_TIMESTAMP 16 // window size
#define SRC_LOC 0 // for dijkstra
#define MAX_LABEL 1000
#define NUM_MC 4
// depends on datatype size also
#define MAX_CACHE_LINES (V/16)
#if TESSERACT == 1
#define LANE_WIDTH 32
#else
#define LANE_WIDTH 1
#endif
#define LINK_BW 16

// things for the network
#define bus_width 32
#define message_size 4

typedef pair<int, int> iPair;

// TODO: add print status everywhere
struct task_id {
  int timestamp;
  int virtual_ind;
  // int core_id; // because multiple cores could be issuing at the same time (not sure how efficient this is)
  task_id() {}
  task_id(int x, int y) { // , int z) {
    timestamp=x; virtual_ind=y; // core_id=z;
  }
};

struct task_entry {
  int vid; // source id
  int start_offset; // to keep track of work left
  int tid;
  task_entry(int x, int y , int a) {
    vid=x; start_offset=y; tid=a;
  }
};

// TODO: make wgt datatype int as a macro
struct edge_info {
  int dst_id;
  int wgt;
  // edge_info(int x, int y) {
  //   dst_id=x; wgt=y;
  // }
};

// task_id is parent task id
struct pref_tuple {
  edge_info edge;
  int src_dist;
  int src_id;
  int tid;
  pref_tuple() {}
  pref_tuple(int a, int b, int z) {
    edge.dst_id=a; edge.wgt=b; tid=z;
  }
};

// task_id is parent task id
struct red_tuple {
  int new_dist;
  int dst_id;
  int src_id;
  int tid;
  int label; // required for network
  int req_core_id; // required for memory accesses
  int dest_core_id; // instead of calculating it every time: set when packet is created and read this only
  int lane_id;
  bool inactive_flag;
  bool local_flag;
  red_tuple() {}
  red_tuple(int x, int y) {
    new_dist=x; dst_id=y; req_core_id=-1;
  }
};

// child_task_id is the new entry to commit queue (it should be the index)
/*struct spec_tuple {
  task_id parent_tid;
  task_id own_tid;
  int finished_left;
  int committed_left;
  int src_id;
  int vid; // commit info (this is same as wr index) (dest_id)
  int new_dist;
  bool spec_flag;
  spec_tuple() {
    finished_left=0;
  }
};*/

struct meta_info {
  int finished_left;
  int committed_left;
  meta_info(int x, int y) {
    finished_left=x; committed_left=y;
  }
};

struct commit_info {
  int vid;
  // int timestamp;
  int parent_tid;
  int own_tid;
  commit_info(int x, int y, int z, int a) {
    vid=x; // timestamp=y; 
    parent_tid=z; own_tid=a;
  }
};

typedef pair<int, commit_info> commit_pair;

#endif
