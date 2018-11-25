#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "scnn.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#define NUM_THREADS	2

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

// Barrier variable
pthread_barrier_t barr;

#define num_inputs (Tx*Ty*Tn)

// uint16_t out_n[Nn][Ty-Ky+1][Tx-Kx+1];
uint16_t out_n[Nn][(Ny-Ky+1)*(Nx-Kx+1)];

// weights -- same thing would go upto Nn/Tn iterations in time
vector<uint16_t> synapse_val[Ni][Nn/Tn];
vector<uint16_t> synapse_ind[Ni][Nn/Tn];
uint16_t synapse_ptr[Nn*Ni/Tn];

// activations -- same thing would go Ni times
vector<uint16_t> neuron_i_val[(Nx*Ny)/(Tx*Ty)][Ni];
vector<uint16_t> neuron_i_ind[(Nx*Ny)/(Tx*Ty)][Ni];
uint16_t neuron_i_ptr[(Nx*Ny*Ni)/(Tx*Ty)];

// sparse output activations (TODO: allocate space for memory for now)
// vector<uint16_t> neuron_o_val[(Nx*Ny)/(Tx*Ty)][Nn];
// vector<uint16_t> neuron_o_ind[(Nx*Ny)/(Tx*Ty)][Nn];
uint16_t neuron_o_val[(Nx*Ny)/(Tx*Ty)][Nn][Tx*Ty];
uint16_t neuron_o_ind[(Nx*Ny)/(Tx*Ty)][Nn][Tx*Ty];

uint16_t neuron_o_ptr[(Nx*Ny*Nn)/(Tx*Ty)];

int count=0;

// This two functions are for a given tid only
void load_weights_in_linear_scratch(int y, int x) {
  unsigned size_synapse = synapse_val[y][x].size();
  SB_DMA_SCRATCH_LOAD(&synapse_val[y][x][0], 2, 2, size_synapse, getLinearAddr(getLinearOffset(0,2)));
  SB_DMA_SCRATCH_LOAD(&synapse_ind[y][x][0], 2, 2, size_synapse, getLinearAddr(getLinearOffset(1,2)));
  // SB_WAIT_SCR_WR();
  SB_WAIT_ALL();
}

// linear scr -> remote port
void broadcast_weights(long tid, int y, int x) {
  uint64_t mask = 0;
  for(int i=0; i<NUM_THREADS; ++i) {
    addDest(mask, i);
  }
  unsigned size_synapse = synapse_val[y][x].size();
  // FIXME: find indirect ports -- hopefully, we don't write from scr to memory here
  SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(0,2)), size_synapse*2, mask, SCR_MEM_PORT);
  SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)), size_synapse*2, mask, P_IND_5);
}

// working on act of neuron_i_val[z][y]
// working on weight of synapse_val[y][x]
void kernel(int x, int y, int z) {

  unsigned size_neuron_tile = neuron_i_val[z][y].size();
  // not sure why ceil is not working
  unsigned ceiled_sn_tile = ceil(size_neuron_tile/float(8));
  // ceiled_sn_tile = ceiled_sn_tile*8;

  if(size_neuron_tile%8!=0) {
	ceiled_sn_tile = size_neuron_tile+8-size_neuron_tile%8;
  } else {
	ceiled_sn_tile = size_neuron_tile;
  }
  int pad_size = ceiled_sn_tile-size_neuron_tile;

  unsigned size_synapse = synapse_val[y][x].size();

  // although this should be taken care in the simulator
  if(size_synapse==0 || size_neuron_tile==0)
	return;

  int num_comp_inst = size_synapse*ceiled_sn_tile;
  // cout << "COMPUTE LENGTHS: ";
  // cout << size_neuron_tile << " " << ceiled_sn_tile << " " << size_synapse << " " << num_comp_inst << endl;

  // SB_CONST_SCR(0, 0, Tx*Ty*Tn);
  // SB_WAIT_ALL();

  // Should change with double buffering if we would be swapping
  int prev_val_scr_offset = getBankedOffset(1,4); 
  int prev_ind_scr_offset = getBankedOffset(2,4);
  int cur_val_scr_offset = getBankedOffset(3,4);
  int cur_ind_scr_offset = getBankedOffset(4,4);

  // frequency of synapses
  // num_elem depends on num_writes
  SB_REPEAT_PORT(ceiled_sn_tile/8);
  SB_RECURRENCE(SCR_MEM_PORT, P_scnn_sval, size_synapse);

  SB_REPEAT_PORT(ceiled_sn_tile/8);
  SB_RECURRENCE(P_IND_5, P_scnn_sind, size_synapse);
 
  SB_CONST(P_scnn_ky, Ky, num_comp_inst/8);
  SB_CONST(P_scnn_ty, Ty, num_comp_inst/8);
  // SB_CONST(P_scnn_const, Ty-Ky+1, num_comp_inst/8);
  SB_CONST(P_scnn_const, Ty-Ky, num_comp_inst/8);

  // frequency of neurons
  for(unsigned is=0; is<size_synapse; ++is) {
    SB_SCRATCH_READ(cur_val_scr_offset, size_neuron_tile*2, P_scnn_nval);
    SB_SCRATCH_READ(cur_ind_scr_offset, size_neuron_tile*2, P_scnn_nind);
 
	SB_CONST(P_scnn_nval, 0, pad_size); // nothing should happen here
	SB_CONST(P_scnn_nind, 1, pad_size);
  }
  SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T32);
  SB_ATOMIC_SCR_OP(P_scnn_A, P_scnn_B, 0, num_comp_inst, 0);

  // re-sparsification code -- let's skip this for the time being
  SB_DMA_READ(&out_n[x*Tn][z*(Kx*Ky)], 2, 2, num_inputs, P_scnn_neuron);
  SB_SCRATCH_READ(prev_val_scr_offset, num_inputs*2, P_scnn_neuron);
  SB_CONST(P_scnn_num_in, num_inputs, num_inputs);
  SB_CONST(P_scnn_acc_ctrl, 0, num_inputs);

  // FIXME: may be some error here (Some error in arbitration)
  // SB_SCR_WRITE(P_scnn_inval, num_inputs*2*2, prev_val_scr_offset);
  // SB_SCR_WRITE(P_scnn_inval, num_inputs*2*2, prev_ind_scr_offset);
 
  SB_STRIDE(2,2);
  SB_DMA_WRITE_SIMP(P_scnn_inval, num_inputs*2, &neuron_o_val[z][x*Tn][0]);
  SB_DMA_WRITE_SIMP(P_scnn_inind, num_inputs*2, &neuron_o_ind[z][x*Tn][0]);

  SB_WAIT_SCR_ATOMIC();
  uint16_t done;
  SB_RECV(P_scnn_done, done);
  SB_RESET();

  SB_WAIT_ALL();
}

// depending on tid chose which core it should go
// FIXME: need different streams because of 2-D pattern
void send_halos(long tid) {
  // this is important to know this address
  int src_val_offset = getBankedOffset(2,4);
  int src_ind_offset = getBankedOffset(3,4);
  int dest_val_offset = 0;  
  int dest_ind_offset = 0;

  if(tid+1 < NUM_THREADS) {
    dest_val_offset = getRemoteAddr(tid+1, getBankedOffset(2,4) + (Tx*Ty*2));
    dest_ind_offset = getRemoteAddr(tid+1, getBankedOffset(3,4) + (Tx*Ty*2));
    SB_SCR_REM_SCR(src_val_offset, 2, 2, (Tx)*2, dest_val_offset, 0);
    SB_SCR_REM_SCR(src_ind_offset, 2, 2, (Tx)*2, dest_ind_offset, 0);
  }
  if(tid-1 > 0 && tid-1 < NUM_THREADS) {
    dest_val_offset = getRemoteAddr(tid-1, getBankedOffset(2,4) + (Tx*Ty*2));
    dest_ind_offset = getRemoteAddr(tid-1, getBankedOffset(3,4) + (Tx*Ty*2));
    SB_SCR_REM_SCR(src_val_offset, Tx*2, 2, (Ty)*2, dest_val_offset+Tx*2, 0);
    SB_SCR_REM_SCR(src_ind_offset, Tx*2, 2, (Ty)*2, dest_ind_offset+Tx*2, 0);
  }
  if(tid+8 < NUM_THREADS) {
    dest_val_offset = getRemoteAddr(tid+8, getBankedOffset(2,4) + (Tx*Ty*2));
    dest_ind_offset = getRemoteAddr(tid+8, getBankedOffset(3,4) + (Tx*Ty*2));
    SB_SCR_REM_SCR(src_val_offset+Tx*(Ty-1)*2, 2, 2, (Tx)*2, dest_val_offset+Tx*2+Ty*2, 0);
    SB_SCR_REM_SCR(src_ind_offset+Tx*(Ty-1)*2, 2, 2, (Tx)*2, dest_ind_offset+Tx*2+Ty*2, 0);
  }
  if(tid-8 > 0 && tid-8 < NUM_THREADS) {
    dest_val_offset = getRemoteAddr(tid-8, getBankedOffset(2,4) + (Tx*Ty*2));
    dest_ind_offset = getRemoteAddr(tid-8, getBankedOffset(3,4) + (Tx*Ty*2));
    SB_SCR_REM_SCR(src_val_offset+Tx*2, Tx*2, 2, (Ty)*2, dest_val_offset+Tx*4+Ty*2, 0);
    SB_SCR_REM_SCR(src_ind_offset+Tx*2, Tx*2, 2, (Ty)*2, dest_ind_offset+Tx*4+Ty*2, 0);
  }
  SB_WAIT_ALL();
}

// TODO: make it general
int halo_count(long tid) {
  if(tid==0) return 0;
  if(tid==1) return (Tx*2);
  return (Ty*2);
}

// FIXME: fix the count thing
void convolution_layer_blocked(long tid) {
  SB_CONFIG(scnn_config,scnn_size);
  int n_count = halo_count(tid);
  int stride = (Nx*Ny)/(Tx*Ty);
  for(int i=0; i<Nn/Tn; ++i) {
	for(int j=0; j<Ni; ++j){
      if(tid==i*Ni+j) {
        load_weights_in_linear_scratch(j,i);
        broadcast_weights(tid, j, i);
        // count++;
      }
	  // all of them use the same weights
	  for(int k=tid*stride; k<stride*(1+tid); ++k) {
		kernel(i,j,k);
	  }
      send_halos(tid);
      SB_WAIT_DF(n_count, 0);
	}
  }
}

void *entry_point(void *threadid) {
  
   long tid;
   tid = (long)threadid;
   // Synchronization point
   int rc = pthread_barrier_wait(&barr);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
   {
     printf("Could not wait on barrier\n");
   }

   begin_roi();
   // double buffering here
   convolution_layer_blocked(tid);
   // for them to execute in parallel, there should have been no wait all (we
   // do not want to stall the control core here)
   // communication(); // will this happen in parallel?
   end_roi();
   sb_stats();
   return NULL;
}

void read_weights() {

  string str(wgt_val_file);
  FILE *weight_val_file = fopen(str.c_str(),"r"); 
  // FILE *weight_val_file = fopen("datasets/wgt_val.data","r"); 
  char lineToRead[5000];
  
  while(fgets(lineToRead, 5000, weight_val_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	float var;

	for(int i=0; i<Ni; ++i){
	  for(int j=0; j<Nn/Tn; ++j){
		iss >> var;
		synapse_val[i][j].push_back(DOUBLE_TO_FIX(var));
	  }
	}
  }
  fclose(weight_val_file);

  str = wgt_ind_file;
  FILE *weight_ind_file = fopen(str.c_str(),"r"); 
  // FILE *weight_ind_file = fopen("datasets/wgt_index.data","r"); 
  
  while(fgets(lineToRead, 5000, weight_ind_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    uint16_t var;

	for(int i=0; i<Ni; ++i){
	  for(int j=0; j<Nn/Tn; ++j){
		iss >> var;
		synapse_ind[i][j].push_back(var);
	  }
	}
	
  }
  fclose(weight_ind_file);

  str = wgt_ptr_file;
  FILE *weight_ptr_file = fopen(str.c_str(),"r"); 
  // FILE *weight_ptr_file = fopen("datasets/wgt_ptr.data","r"); 
  
  while(fgets(lineToRead, 5000, weight_ptr_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<Ni*Nn/Tn; ++i){
	  iss >> synapse_ptr[i];
	}
  }
  fclose(weight_ptr_file);
}

void read_activations() {

  int dim1 = (Nx*Ny)/(Tx*Ty);

  string str(act_val_file);
  FILE *act_val = fopen(str.c_str(),"r"); 
  // FILE *act_val = fopen("datasets/act_val.data","r"); 
  char lineToRead[5000];

  while(fgets(lineToRead, 5000, act_val)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	float var;

	for(int i=0; i<dim1; ++i){
	  for(int j=0; j<Ni; ++j){
		iss >> var;
		neuron_i_val[i][j].push_back(DOUBLE_TO_FIX(var));
	  }
	}
  }
  fclose(act_val);

  str = act_ind_file;
  FILE *act_ind = fopen(str.c_str(),"r"); 
  // FILE *act_ind = fopen("datasets/act_index.data","r"); 
  
  while(fgets(lineToRead, 5000, act_ind)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    uint16_t var;

	for(int i=0; i<dim1; ++i){
	  for(int j=0; j<Ni; ++j){
		iss >> var;
		neuron_i_ind[i][j].push_back(var);
	  }
	}
  }
  fclose(act_ind);

  str = act_ptr_file;
  FILE *act_ptr = fopen(str.c_str(),"r"); 
  // FILE *act_ptr = fopen("datasets/act_ptr.data","r"); 
  
  while(fgets(lineToRead, 5000, act_ptr)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<dim1*Ni; ++i){
	  iss >> neuron_i_ptr[i];
	}
  }
  fclose(act_ptr);
}


void fill_convolution_data() {

  read_weights();
  cout << "Done reading weights\n";

  read_activations();
  cout << "Done reading activations\n";

  // FILE *out_neuron_file = fopen("output_neuron.data","r"); 
  // char lineToRead[5000];
  // int t=0; float t2;
  // union a temp;
  // // int16_t out_n[num_inputs];
  // printf("Started reading output neuron file!\n");
  // while(fgets(lineToRead, 5000, out_neuron_file)!=NULL){
  //   std::string raw(lineToRead);
  //   std::istringstream iss(raw.c_str());

  //   for(int i=0; i<Tx*Tx; ++i){
  //     iss >> temp.x;
  //     out_n[t*Tn+i] = temp.y;
  //   }
  //   t++;
  // }

  // out_n[num_inputs-1] = SENTINAL16;

  printf("Done reading file!\n");

}


int main() {

  cout << "initializing arrays\n";

  fill_convolution_data();

  cout << "starting computation\n";

  /*
  begin_roi();
  convolution_layer_blocked(0);
  end_roi();
  sb_stats();
  */

  assert(NUM_THREADS<C);
  
  // Barrier initialization
  if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  {
    printf("Could not create a barrier\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  int rc;
  long t;
  for(t=0;t<NUM_THREADS;t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);     
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      return 0;
    }
  }
  
  for(int i = 0; i < NUM_THREADS; ++i) {
    if(pthread_join(threads[i], NULL)) {
  	printf("Could not join thread %d\n", i);
      return -1;
    }
  }

  cout << "blocked computation complete!\n";  

  // compare((uint16_t*)*neuron_n,(uint16_t*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
  return 0;
}

/*
void broadcast_weights(long tid, int y, int x) {
  // linear scr -> remote port

  unsigned size_synapse = synapse_val[y][x].size();
  // Either make these ports 1-byte or use some other ports, FIXME: need
  // padding here also
  // scr_mem_port, scr_rem_port are free
  // SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)), size_synapse, mask, P_IND_1);
  // SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(2,2)), size_synapse, mask, P_IND_2);

  // FIXME: cannot use this because the destination can be local also (let's
  // keep that separate) -- anyways routers should not be crowded
  // FOR REMOTE
  uint64_t mask = 0; // derive from here
  for(int i=0; i<NUM_THREADS; ++i) {
    if(tid!=i) addDest(mask, i);
  }
  if(mask!=0) { // should be there in simulator itself
    SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)), size_synapse, mask, SCR_MEM_PORT);
    SB_SCR_REM_PORT(getLinearAddr(getLinearOffset(2,2)), size_synapse, mask, SCR_REM_PORT);
  }

  // FOR LOCAL (multiple reads -- improve)
  SB_SCRATCH_READ(getLinearAddr(getLinearOffset(1,2)), 

}
*/
