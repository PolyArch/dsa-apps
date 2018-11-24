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

// #define Ni 96
// #define Nx 55
// #define Ny 55
// // TODO: this should be made balanced when done across cores
// #define Tx 7
// #define Ty 7
// #define Ni 96
// #define Nn 256
// // #define Tn 256
// #define Tn 64
// #define Kx 5
// #define Ky 5
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
  cout << size_neuron_tile << " " << ceiled_sn_tile << " " << size_synapse << " " << num_comp_inst << endl;

  SB_CONST_SCR(0, 0, Tx*Ty*Tn);
  SB_WAIT_ALL();

  SB_CONFIG(scnn_config,scnn_size);

  // frequency of synapses
  SB_REPEAT_PORT(ceiled_sn_tile/8);
  SB_DMA_READ(&synapse_val[y][x][0], 2, 2, size_synapse, P_scnn_sval);

  SB_REPEAT_PORT(ceiled_sn_tile/8);
  SB_DMA_READ(&synapse_ind[y][x][0], 2, 2, size_synapse, P_scnn_sind);
 
  SB_CONST(P_scnn_ky, Ky, num_comp_inst/8);
  SB_CONST(P_scnn_ty, Ty, num_comp_inst/8);
  SB_CONST(P_scnn_const, Ty-Ky+1, num_comp_inst/8);

  // frequency of neurons
  for(unsigned is=0; is<size_synapse; ++is) {
    SB_DMA_READ(&neuron_i_val[z][y][0], 2, 2, size_neuron_tile, P_scnn_nval);
    SB_DMA_READ(&neuron_i_ind[z][y][0], 2, 2, size_neuron_tile, P_scnn_nind);
	SB_CONST(P_scnn_nval, 0, pad_size); // nothing should happen here
	SB_CONST(P_scnn_nind, 1, pad_size);
  }
  SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T32);
  SB_ATOMIC_SCR_OP(P_scnn_A, P_scnn_B, 0, num_comp_inst, 0);

  // re-sparsification code
  SB_DMA_READ(&out_n[x*Tn][z*(Kx*Ky)], 2, 2, num_inputs, P_scnn_neuron);
  SB_CONST(P_scnn_num_in, num_inputs, num_inputs);
  SB_CONST(P_scnn_acc_ctrl, 0, num_inputs);

  SB_STRIDE(2,2);
  SB_DMA_WRITE_SIMP(P_scnn_inval, num_inputs*2, &neuron_o_val[z][x*Tn][0]);
  SB_DMA_WRITE_SIMP(P_scnn_inind, num_inputs*2, &neuron_o_ind[z][x*Tn][0]);

  SB_WAIT_SCR_ATOMIC();
  uint64_t done;
  SB_RECV(P_scnn_done, done);
  SB_RESET();


  SB_WAIT_ALL();
}
/*
void send_halos() {
  // this is important to know this address
  SB_SCR_REM_SCR(0, 8, 8, (Tx*Ty*4)*2/8, 0, 0);
}
*/


void convolution_layer_blocked(long tid) {
  // wait on the halos
  // SB_WAIT_DF((Tx*Ty*4)*2, 0);
  int stride = (Nx*Ny)/(Tx*Ty);
  for(int i=0; i<Nn/Tn; ++i) {
	for(int j=0; j<Ni; ++j){
	  // all of them use the same weights
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
