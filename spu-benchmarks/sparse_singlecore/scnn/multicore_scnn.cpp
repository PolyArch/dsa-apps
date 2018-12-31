#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "scnn.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/net_util_func.h"
#include "/home/vidushi/ss-stack/ss-scheduler/src/config/fixed_point.h"
#include <inttypes.h>
#include <assert.h>
#define NUM_THREADS	2

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif


// #define Kxsim (Kx | Kx << 16 | (Kx & 0xFFFFFFFFFFFFFFFF) << 32 | (Kx & 0xFFFFFFFFFFFFFFFF) << 48)
// #define Nxsim (Nx | Nx << 16 | (Nx & 0xFFFFFFFFFFFFFFFF) << 32 | (Nx & 0xFFFFFFFFFFFFFFFF) << 48)
// #define Txsim (Tx | Tx << 16 | (Tx & 0xFFFFFFFFFFFFFFFF) << 32 | (Tx & 0xFFFFFFFFFFFFFFFF) << 48)

#define fused_const (1 | 1 << 16 | (1 & 0xFFFFFFFFFFFFFFFF) << 32 | (1 & 0xFFFFFFFFFFFFFFFF) << 48)
#define SCRATCH_SIZE 16384

// Barrier variable
pthread_barrier_t barr;

/*
// 3rd dim in weights
for(int x=0; x<Nn/Tn; x++) {
  // 2nd dim in both 
  for(int y=0; y<Ni; y++) {
	// 1 dim of sparse input act ---------------------------
	for(int w=tid*Tx; w<(tid*Tx+Tx); w++){
     for(int w=tid*Tx; w<(tid*Tx+Tx); w++){
	   // -------------------------
	   // 1-dim of sparse weights --------------------
	   for(int t=0; t<Tn; t++){
	   for(int r=0; r<Kx; r++){
		 for(int s=0; s<Ky; s++){

		}
	  }
	   }
	   // -------------------
	 }  
	}
	
  }
  scratch_move();
}

// in stream format
// 3rd dim in weights
for(int x=0; x<Nn/Tn; x++) {
  // 2nd dim in both 
  for(int y=0; y<Ni; y++) {
	// inter-iteration reuse
	for(int i=0; i < n_times; i++) {
	  stream_load(input_act[tid*Tx], Tx);
	}
	repeat();
	stream_load(weights[x*Kx*Ky], Tn*Kx*Ky);
  }
  scratch_move();
}
*/

#define Ni 96
#define Nx 55
#define Ny 55
// TODO: this should be made balanced when done across cores
#define Tx 7
#define Ty 7
#define Ni 96
#define Nn 256
// #define Tn 256
#define Tn 64
#define Kx 5
#define Ky 5
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

// sparse output activations
vector<uint16_t> neuron_o_val[(Nx*Ny)/(Tx*Ty)][Nn];
vector<uint16_t> neuron_o_ind[(Nx*Ny)/(Tx*Ty)][Nn];
uint16_t neuron_o_ptr[(Nx*Ny*Nn)/(Tx*Ty)];


int count=0;

void load_weights_in_linear_scratch(int y, int x) {

  unsigned size_synapse = synapse_val[y][x].size();
  SS_DMA_SCRATCH_LOAD(&synapse_val[y][x][0], 2, 2, size_synapse, getLinearAddr(getLinearOffset(1,2)));
  SS_DMA_SCRATCH_LOAD(&synapse_ind[y][x][0], 2, 2, size_synapse, getLinearAddr(getLinearOffset(2,2)));
  // SS_WAIT_SCR_WR();
  count++;
}

void broadcast_weights(unsigned size_synapse) {
  // linear scr -> remote port
  uint64_t mask = SENTINAL; // derive from here
  SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(1,2)), 2, 2, size_synapse, mask, P_IND_1);
  SS_SCR_REM_PORT(getLinearAddr(getLinearOffset(2,2)), 2, 2, size_synapse, mask, P_IND_2);
}

// working on act of neuron_i_val[z][y]
// working on weight of synapse_val[y][x]
void kernel(int x, int y, int z) {

  unsigned size_neuron_tile = neuron_i_val[z][y].size();
  unsigned size_synapse = synapse_val[y][x].size();
  int num_comp_inst = size_synapse*size_neuron_tile; // number of 16-bit instances required

  // FIXME: problem when dense does not fit in scratch (should use linear/dma
  // for that)
  // Assuming we work on only 1 layer (swapping of double buffer is not emulated yet)
  int prev_val_scr_offset = getBankedOffset(1,4); // TODO: assumed same size?
  int prev_ind_scr_offset = getBankedOffset(2,4); // TODO: assumed same size?
  int cur_val_scr_offset = getBankedOffset(3,4); // same size?
  int cur_ind_scr_offset = getBankedOffset(4,4); // same size?

  SS_CONST_SCR(0, 0, Tx*Ty*Tn);
  SS_WAIT_ALL();
  // SS_WAIT_SCR_WR();

  SS_CONFIG(scnn_config,scnn_size);

  // re-sparsification code
  // TODO: should read from banked scratchpad
  // FIXME: check the initial value
  // SS_DMA_READ(&out_n[x*Tn][z*(Kx*Ky)], 2, 2, num_inputs, P_scnn_neuron);
  SS_SCRATCH_READ(prev_val_scr_offset, num_inputs*2, P_scnn_neuron);
  SS_CONST(P_scnn_constt, fused_const, num_inputs);
  SS_CONST(P_scnn_dummy, 1, num_inputs);
  // TODO: should write to? maybe memory here (but ideally?)
  // Write to banked scratchpad addresses otherwise malloc required (overwrite
  // over the previous one because result cannot be generated unless it is read
  // -- so only 2 partitions should be fine)
  SS_SCR_WRITE(P_scnn_inval, num_inputs*2*8, prev_val_scr_offset);
  SS_SCR_WRITE(P_scnn_inval, num_inputs*2*8, prev_ind_scr_offset);
  // SS_DMA_WRITE_SIMP(P_scnn_inval, num_inputs*2, &neuron_o_val[z][x*Tn][0]);
  // SS_DMA_WRITE_SIMP(P_scnn_inind, num_inputs*2, &neuron_o_ind[z][x*Tn][0]);

  // convolution code 
  SS_REPEAT_PORT(size_neuron_tile);
  SS_RECURRENCE(P_IND_1, P_scnn_sval, size_synapse);

  SS_REPEAT_PORT(size_neuron_tile);
  SS_RECURRENCE(P_IND_2, P_scnn_sind, size_synapse);

  // TODO: should be coming from banked scratchpad
  // for(unsigned is=0; is<size_synapse; ++is) {
  //   SS_DMA_READ(&neuron_i_val[x][y][0], sizeof(uint16_t), sizeof(uint16_t), neuron_i_val[x][y].size(), P_scnn_nval);
  //   SS_DMA_READ(&neuron_i_ind[x][y][0], sizeof(uint16_t), sizeof(uint16_t), neuron_i_ind[x][y].size(), P_scnn_nval);
  // }
  for(unsigned is=0; is<size_synapse; ++is) {
    SS_SCRATCH_READ(cur_val_scr_offset, neuron_i_val[x][y].size()*2, P_scnn_nval);
    SS_SCRATCH_READ(cur_ind_scr_offset, neuron_i_val[x][y].size()*2, P_scnn_nind);
  }
 
  SS_CONST(P_scnn_Ky, Ky, num_comp_inst);
  SS_CONST(P_scnn_Ty, Ty, num_comp_inst);
  SS_CONST(P_scnn_const, Ty-Ky+1, num_comp_inst);

  SS_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SS_ATOMIC_SCR_OP(P_scnn_C, P_scnn_D, 0, num_comp_inst, 0);

  //----------------------------
  SS_WAIT_SCR_ATOMIC();
  uint64_t done;
  SS_RECV(P_scnn_done, done);
  SS_RESET();
  SS_WAIT_ALL();
}

void send_halos(long tid) {
  // this is important to know this address
  // TODO: make scr_scr_port also 8-bit wide
  // for the first linear line, TODO: check this location
  SS_SCR_REM_SCR(getBankedOffset(3,4), 8, 8, (Tx)*2/8, getRemoteBankedOffset(tid+1, 4, 4)+(Tx*Ty*2), 0);
  SS_SCR_REM_SCR(getBankedOffset(4,4), 8, 8, (Tx)*2/8, getRemoteBankedOffset(tid+1, 4, 4)+(Tx*Ty*2)+(Ty)*2, 0);
  // TODO: add for other boundaries
}

void convolution_layer_blocked(long tid) {
  // wait on the halos
  // SS_WAIT_DF((Tx*Ty*4)*2, 0);
  load_weights_in_linear_scratch(0,0);
  int stride = (Nx*Ny)/(Tx*Ty);
  for(int i=0; i<Nn/Tn; ++i) {
	for(int j=0; j<Ni; ++j){
	  // all of them use the same weights
      SS_WAIT_SCR_WR();
	  if(j+1<Ni && tid==count) load_weights_in_linear_scratch(j+1, i);
	  if(tid==count) broadcast_weights(synapse_val[j][i].size());
	  for(int k=tid*stride; k<stride*(1+tid); ++k) {
		kernel(i,j,k);
	  }
	}
  }
  // send_halos();
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

  FILE *weight_val_file = fopen("datasets/wgt_val.data","r"); 
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

  FILE *weight_ind_file = fopen("datasets/wgt_index.data","r"); 
  
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

  FILE *weight_ptr_file = fopen("datasets/wgt_ptr.data","r"); 
  
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

  FILE *act_val_file = fopen("datasets/act_val.data","r"); 
  char lineToRead[5000];

  while(fgets(lineToRead, 5000, act_val_file)!=NULL){
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
  fclose(act_val_file);

  FILE *act_ind_file = fopen("datasets/act_index.data","r"); 
  
  while(fgets(lineToRead, 5000, act_ind_file)!=NULL){
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
  fclose(act_ind_file);

  FILE *act_ptr_file = fopen("datasets/act_ptr.data","r"); 
  
  while(fgets(lineToRead, 5000, act_ptr_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<dim1*Ni; ++i){
	  iss >> neuron_i_ptr[i];
	}
  }
  fclose(act_ptr_file);
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

  begin_roi();
  convolution_layer_blocked(0);
  end_roi();
  sb_stats();

  // assert(NUM_THREADS<C);
  // 
  // // Barrier initialization
  // if(pthread_barrier_init(&barr, NULL, NUM_THREADS))
  // {
  //   printf("Could not create a barrier\n");
  //   return -1;
  // }

  // pthread_t threads[NUM_THREADS];
  // int rc;
  // long t;
  // for(t=0;t<NUM_THREADS;t++){
  //   printf("In main: creating thread %ld\n", t);
  //   rc = pthread_create(&threads[t], NULL, entry_point, (void *)t);     
  //   if (rc){
  //     printf("ERROR; return code from pthread_create() is %d\n", rc);
  //     return 0;
  //   }
  // }
  // 
  // for(int i = 0; i < NUM_THREADS; ++i) {
  //   if(pthread_join(threads[i], NULL)) {
  // 	printf("Could not join thread %d\n", i);
  //     return -1;
  //   }
  // }

  // cout << "blocked computation complete!\n";  

  // compare((uint16_t*)*neuron_n,(uint16_t*)*neuron_n2,NYSCL*NXSCL*Nn);

  // cout << "done\n";
  return 0;
}

