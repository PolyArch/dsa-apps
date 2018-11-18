#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "scnn.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
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


// TODO:FIXME: Check if we should include Ni variation inside the kernel
void kernel(int x, int y, int z) {

  // working on act of neuron_i_val[z][y]
  // working on weight of synapse_val[y][x]

  unsigned size_neuron_tile = neuron_i_val[z][y].size();
  unsigned size_synapse = synapse_val[y][x].size();
  int num_comp_inst = size_synapse*size_neuron_tile; // number of 16-bit instances required

  SB_CONST_SCR(0, 0, Tx*Ty*Tn);
  SB_WAIT_ALL();
  // SB_WAIT_SCR_WR();

  SB_CONFIG(scnn_config,scnn_size);

  // re-sparsification code
  // TODO: should read from banked scratchpad
  // FIXME: check the initial value
  SB_DMA_READ(&out_n[x*Tn][z*(Kx*Ky)], 2, 2, num_inputs, P_scnn_neuron);
  SB_CONST(P_scnn_constt, fused_const, num_inputs);
  SB_CONST(P_scnn_dummy, 1, num_inputs);
  // TODO: should write to? maybe memory here (but ideally?)
  // Write to banked scratchpad addresses otherwise malloc required (overwrite
  // over the previous one because result cannot be generated unless it is read
  // -- so only 2 partitions should be fine)
  SB_DMA_WRITE_SIMP(P_scnn_inval, num_inputs*2, &neuron_o_val[z][x*Tn][0]);
  SB_DMA_WRITE_SIMP(P_scnn_inind, num_inputs*2, &neuron_o_ind[z][x*Tn][0]);

  // convolution code -- TODO: should be coming from banked scratchpad
  SB_REPEAT_PORT(size_neuron_tile);
  SB_DMA_READ(&synapse_val[y][x][0], sizeof(uint16_t), sizeof(uint16_t), size_synapse, P_scnn_sval);

  SB_REPEAT_PORT(size_neuron_tile);
  SB_DMA_READ(&synapse_ind[y][x][0], sizeof(uint16_t), sizeof(uint16_t), size_synapse, P_scnn_sind);

  // printf("size_synapse: %d size_neuron: %d num_comp_inst: %d\n",size_synapse,size_neuron_tile, num_comp_inst);
  
  // TODO: should be coming from linear scratchpad
  for(unsigned is=0; is<size_synapse; ++is) {
    SB_DMA_READ(&neuron_i_val[x][y][0], sizeof(uint16_t), sizeof(uint16_t), neuron_i_val[x][y].size(), P_scnn_nval);
    SB_DMA_READ(&neuron_i_ind[x][y][0], sizeof(uint16_t), sizeof(uint16_t), neuron_i_ind[x][y].size(), P_scnn_nval);
  }
 
  SB_CONST(P_scnn_Ky, Ky, num_comp_inst);
  SB_CONST(P_scnn_Ty, Ty, num_comp_inst);
  SB_CONST(P_scnn_const, Ty-Ky+1, num_comp_inst);

  SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  // SB_CONFIG_ATOMIC_SCR_OP(T16, T32, T16);
  SB_ATOMIC_SCR_OP(P_scnn_C, P_scnn_D, 0, num_comp_inst, 0);

  //----------------------------
  SB_WAIT_SCR_ATOMIC();
  uint64_t done;
  SB_RECV(P_scnn_done, done);
  SB_RESET();
  SB_WAIT_ALL();
}

void convolution_layer_blocked() {
  for(int i=0; i<Nn/Tn; ++i) {
	for(int j=0; j<Ni; ++j){
	  for(int k=0; k<(Nx*Ny)/(Tx*Ty); ++k) {
		kernel(i,j,k);
	  }
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

   // begin_roi();
   // // call function here (some offset determined from threadid)
   // end_roi();
   // sb_stats();
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
  convolution_layer_blocked();
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
