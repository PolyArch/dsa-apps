#include <iostream>
#include <string>
#include <sstream>
#include "scnn.dfg.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sb_insts.h"
#include "/home/vidushi/ss-stack/ss-workloads/common/include/sim_timing.h"
#include <inttypes.h>
#include <assert.h>
#define NUM_THREADS	2

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#define VTYPE uint16_t

#define Kx 3
#define Ky 3
#define Nx 224
#define Ny 224
// #define Nn 64
// #define Nn 1
#define Nn 1

#define Kxsim (Kx | Kx << 16 | (Kx & 0xFFFFFFFFFFFFFFFF) << 32 | (Kx & 0xFFFFFFFFFFFFFFFF) << 48)
#define Nxsim (Nx | Nx << 16 | (Nx & 0xFFFFFFFFFFFFFFFF) << 32 | (Nx & 0xFFFFFFFFFFFFFFFF) << 48)
#define Txsim (Tx | Tx << 16 | (Tx & 0xFFFFFFFFFFFFFFFF) << 32 | (Tx & 0xFFFFFFFFFFFFFFFF) << 48)

#define tile_factor 1


#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Ni)

// #define sparsity_n 0.01
// #define sparsity_n 0.5
// #define sparsity_s 0.55
// #define nnz1 SYNAPSE_SIZE*sparsity_s
#define nnz_syn (synapse_sp*Kx*Ky*Tn)
#define nnz1 (synapse_sp*Kx*Ky*Tn)
// #define nnz2 NYPAD*NXPAD*Ni*sparsity_n
#define nnz_ne (neuron_sp*Tx*Tx*Ni)
#define nnz2 (25076*Ni)

#define fused_const (1 | 1 << 16 | (1 & 0xFFFFFFFFFFFFFFFF) << 32 | (1 & 0xFFFFFFFFFFFFFFFF) << 48)
#define num_inputs (Tx*Tx*Tn)

// Barrier variable
pthread_barrier_t barr;

// GLOBAL DATA STRUCTURES HERE
int16_t (*out_n)[num_inputs];
 
VTYPE (*synapse_val)[Ni][int(nnz1)];
VTYPE (*synapse_ind)[Ni][int(nnz1)];
VTYPE (*synapse_ptr)[Nn][Ni+1];
VTYPE (*neuron_i_val)[int(nnz2)];
VTYPE (*neuron_i_ind)[int(nnz2)];
VTYPE (*neuron_ptr)[Ni*tile_factor+1];
VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];



union a{
  float x;
  int16_t y;
};

void fill_convolution_data(VTYPE (&synapse_val)[Ni][int(nnz1)],VTYPE (&synapse_ind)[Ni][int(nnz1)],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor+1], VTYPE (&synapse_ptr)[Nn][Ni+1], int16_t (&out_n)[num_inputs]) {
// void fill_convolution_data() {
  char lineToRead[5000];
  int id=0;
  // FILE* neuron_file = fopen("dataset_rle.txt","r");
  FILE* neuron_file = fopen("input_neuron.data","r");
  while(fgets(lineToRead, 5000, neuron_file) != NULL) {
	// this is the number of values in the rle_width
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	char ignore;

	for(int i=0; i<rle_width; ++i){
	  float a,b;
	  iss >> a >> b;
	  // std::cout << a << " " << b << "\n";
	  // neuron_i_ind[id] = (int16_t)(b * (1 << 8));
	  neuron_i_ind[id] = (uint16_t)(b);
	  neuron_i_val[id] = (uint16_t)(a * (1 << 8));
	  // iss >> neuron_i_val[id] >> neuron_i_ind[id];
	  
	  // std::cout << neuron_i_val[id] << " " << neuron_i_ind[id] << "\n";
	  id++;
	}
  }

  // for neuron: id is nnz2 now
  for(int i = 0; i < tile_factor*Ni; ++i) {
     neuron_ptr[i] = (id*i)/(tile_factor*Ni);
  }
  neuron_ptr[Ni*tile_factor] = id;

  // nnz1 = 4*Nn*Ni;
  // srand(1);
  // synapse_val[Ni][Kx*Ky*Tn];
  id=0;
  FILE* synapse_file = fopen("input_synapse.data","r");
  // this data is being generated for all Ni
  while(fgets(lineToRead, 5000, synapse_file) != NULL) {

	// std::string raw(lineToRead);
	// std::istringstream iss(raw.c_str());
	// float a,b;
	// iss >> a >> b;

	// synapse_ind[0][id] = (uint16_t)(b);
	// synapse_val[0][id] = (uint16_t)(a * (1 << 8));
	  
	// std::cout << synapse_val[0][id] << " " << synapse_ind[0][id] << "\n";

    sscanf(lineToRead, "%hu %hu", &synapse_val[0][id], &synapse_ind[0][id]);
    id++;
  }
  // duplicate for Ni feature maps
  for(int j = 1; j < Ni; ++j) {
    for(int i = 0; i < id; ++i) {
      synapse_val[j][i] = synapse_val[0][i];
      synapse_ind[j][i] = synapse_ind[0][i];
    }
  }

  FILE *out_neuron_file = fopen("output_neuron.data","r"); 
  int t=0; float t2;
  union a temp;
  // int16_t out_n[num_inputs];
  printf("Started reading output neuron file!\n");
  while(fgets(lineToRead, 5000, out_neuron_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<Tx*Tx; ++i){
	  iss >> temp.x;
	  out_n[t*Tn+i] = temp.y;
	}
	t++;
  }

  out_n[num_inputs-1] = SENTINAL16;

  printf("Done reading file!\n");

}


void initialize_output_neuron(int16_t (&out_n)[num_inputs]){

  // int num_inputs = Tx*Tx*Tn;
  FILE *out_neuron_file = fopen("output_neuron.data","r"); 
  int t=0; float t2;
  union a temp;
  // int16_t out_n[num_inputs];
  char lineToRead[5000];
  printf("Started reading file!\n");
  while(fgets(lineToRead, 5000, out_neuron_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());

	for(int i=0; i<Tx*Tx; ++i){
	  iss >> temp.x;
	  out_n[t*Tn+i] = temp.y;
	}
	t++;
  }

  out_n[num_inputs-1] = SENTINAL16;

  printf("Done reading file!\n");
}


// std::pair<int,int> convolution_layer_blocked((VTYPE (&synapse_val)[nnz1][Nn],VTYPE (&synapse_ind)[nnz1][Nn],
void convolution_layer_blocked(VTYPE (&synapse_val)[Ni][int(nnz1)], VTYPE (&synapse_ind)[Ni][int(nnz1)],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor+1], VTYPE (&synapse_ptr)[Nn][Ni+1],
                                   VTYPE (&neuron_n)[NYSCL][NXSCL][Nn], int16_t (&out_n)[num_inputs]) {
  // int c1=0,c2=0;
  // VTYPE sum[Nn]={0};
  int size_neuron_tile = 0;
  int size_synapse = 0;
  int num_comp_inst = 0;

  uint64_t a; uint64_t b[Tn];
  for (int nn = 0; nn < Tn; nn++) {
    a = (Nx + Kx + nn*Tx*Tx);
    b[nn] = (a | a << 16 | (a & 0xFFFFFFFFFFFFFFFF) << 32 | (a & 0xFFFFFFFFFFFFFFFF) << 48);
  }

  VTYPE sy_ind1, sy_ind2, sy_ind3, sy_ind4;
  VTYPE offset1, offset2, offset3, offset4;
  uint64_t weightsim;
  uint64_t valsim;
  uint16_t sy_val;
  int tile_no = 0;

  size_synapse = nnz_syn;
  size_neuron_tile = nnz_ne/Ni; // size of 1 tile
  size_synapse = (size_synapse/4)*4;
  size_neuron_tile = (size_neuron_tile/4)*4;
  // 16 values done every cycle: so inst = prod/16
  num_comp_inst = (size_synapse*size_neuron_tile)/(4*4);

  uint16_t n_val[num_inputs];
  uint16_t n_ind[num_inputs];


  // begin_roi();
  SB_CONST_SCR(0, 0, Tx*Tx*Tn);
  SB_WAIT_ALL();
  // SB_WAIT_SCR_WR();
  // SB_WAIT_SCR_RD();

  SB_CONFIG(scnn_config,scnn_size);

  // re-sparsification phase code
  SB_REPEAT_PORT(4);
  SB_DMA_READ(&out_n[0], 8, 8, num_inputs/4, P_scnn_neuron);
  SB_CONST(P_scnn_constt, fused_const, num_inputs);
  SB_CONST(P_scnn_dummy, 1, num_inputs);
  SB_DMA_WRITE_SIMP(P_scnn_inval, num_inputs*2, &n_val[0]);
  SB_DMA_WRITE_SIMP(P_scnn_inind, num_inputs*2, &n_ind[0]);


  SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
  SB_ATOMIC_SCR_OP(P_scnn_C, P_scnn_D, 0, (size_synapse*size_neuron_tile*Ni), 0);


  SB_CONST(P_scnn_Kx, Kxsim, num_comp_inst*Ni);
  SB_CONST(P_scnn_Tx, Txsim, num_comp_inst*Ni);

  SB_REPEAT_PORT(size_neuron_tile/4);
  SB_DMA_READ(&synapse_val[0][0], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_synapse*Ni/4, P_scnn_sval);

  SB_REPEAT_PORT(size_neuron_tile/4);
  SB_DMA_READ(&synapse_ind[0][0], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_synapse*Ni/4, P_scnn_sind);

  SB_2D_CONST(P_scnn_const, 0, size_neuron_tile/4-1, 1, 1, size_synapse*Ni/4);

  for (int feature_map_id = 0; feature_map_id < Ni; ++feature_map_id) {
   	// printf("size_synapse: %d size_neuron: %d num_comp_inst: %d\n",size_synapse,size_neuron_tile, num_comp_inst);
    int nval_st = neuron_ptr[feature_map_id];
    
	for(int is=0; is<size_synapse/4; ++is) {
      SB_DMA_READ(&neuron_i_val[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_scnn_nval);
      SB_DMA_READ(&neuron_i_ind[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_scnn_nind);
    }
    
	SB_REPEAT_PORT(num_comp_inst/Tn);
	SB_DMA_READ(&b[0], 8, 8, Tn, P_scnn_rle_const);
  }
  SB_WAIT_SCR_ATOMIC();
  uint64_t done;
  SB_RECV(P_scnn_done, done);
  SB_RESET();
  SB_WAIT_ALL();
  // end_roi();
  // sb_stats();
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

int main() {

  // int num_inputs = Tx*Ty*Tn;
  out_n   = (int16_t (*)[num_inputs])  malloc(num_inputs*sizeof(int16_t));
  // initialize_output_neuron(*out_n);
  cout << "allocating memory\n";

  synapse_val   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ind   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ptr        = (VTYPE (*)[Nn][Ni+1])  malloc(Nn*(Ni+1)*sizeof(VTYPE));
  neuron_i_val  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_i_ind  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_ptr        = (VTYPE (*)[Ni*tile_factor+1])  malloc((Ni+1)*tile_factor*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_data(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *out_n);
  // fill_convolution_data();
  // fill_convolution_data(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr);
  // modify_encoding(*synapse_ind,*neuron_i_ind);

  cout << "starting computation\n";

  // convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *neuron_n, out_n, num_inputs);
  // convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *neuron_n);
  // Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *neuron_n, *out_n);
  end_roi();
  sb_stats();

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

  // compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  // cout << "done\n";
  return 0;
}