#include <iostream>
#include <string>
#include "test.dfg.h"
#include "../../common/include/sb_insts.h"
#include "../../common/include/sim_timing.h"
#include <inttypes.h>

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

/*
#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif
*/

#define VTYPE uint16_t

#define Kx 3
#define Ky 3
#define Nx 224
#define Ny 224
// #define Nn 64
#define Nn 36

#define Kxsim (Kx | Kx << 16 | (Kx & 0xFFFFFFFFFFFFFFFF) << 32 | (Kx & 0xFFFFFFFFFFFFFFFF) << 48)
#define Nxsim (Nx | Nx << 16 | (Nx & 0xFFFFFFFFFFFFFFFF) << 32 | (Nx & 0xFFFFFFFFFFFFFFFF) << 48)

// #define Ni 64
// #define tile_factor 8
#define Ni 8
#define tile_factor 8
#define Tn 1 // for now


// #define NYPAD (Ny+Ky)
// #define NXPAD (Nx+Kx)

#define NYPAD (Ny)
#define NXPAD (Nx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Ni)

// #define sparsity_n 0.01
#define sparsity_n 0.5
#define sparsity_s 0.22
#define nnz1 SYNAPSE_SIZE*sparsity_s
#define nnz2 NYPAD*NXPAD*Ni*sparsity_n


void fill_convolution_data(VTYPE (&synapse_val)[int(nnz1)][Nn],VTYPE (&synapse_ind)[int(nnz1)][Nn],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor], VTYPE (&synapse_ptr)[Ni]) {


  for(int i = 0; i < nnz1; ++i) {
    for(int j = 0; j < Nn; ++j) {
          synapse_val[i][j] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (500);
          // synapse_val[i][j] = static_cast <VTYPE> (rand() % 500);
          // synapse_ind[i][j] = static_cast <VTYPE> (rand() % 15);
          synapse_ind[i][j] = static_cast <VTYPE> (rand()%2);
          // synapse_ind[i][j] = 0;
          // synapse_ind[i][j] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (15);
  } }
  // for neuron
  for(int i = 0; i < tile_factor*Ni; ++i) {
      // for now: considering similar sparsity in all feature maps
     // neuron_ptr[i] = (nnz2*(i+1))/(tile_factor*Ni);
     // neuron_ptr[i] = (nnz2*(i+1))/100;
     neuron_ptr[i] = (nnz2*(i+1))/200;
  }
  // for synapse
  for(int i = 0; i < Ni; ++i) {
     // synapse_ptr[i] = (nnz1*(i+1))/(Ni);
     // synapse_ptr[i] = (nnz1*(i+1))/100;
     synapse_ptr[i] = (nnz1*(i+1))/200;
  }
  for(int i = 0; i < nnz2; ++i) {
       neuron_i_val[i] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (500);
       // neuron_i_ind[i] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (15);
       // neuron_i_val[i] = static_cast <VTYPE> (rand()%500);
       // neuron_i_ind[i] = static_cast <VTYPE> (rand()%15);
       neuron_i_ind[i] = static_cast <VTYPE> (rand()%2);
       // neuron_i_ind[i] = 1;
       // neuron_i_ind[i] = 0;

  }
}

void modify_encoding(VTYPE (&synapse_ind)[int(nnz1)][Nn],VTYPE (&neuron_i_ind)[int(nnz2)]) {
  for(int i = 0; i < nnz1; i+=4) {
    for(int j = 0; j < Nn; ++j) {
        for(int k=i+1; k<i+4; ++k) {
           synapse_ind[k][j] += synapse_ind[k-1][j]; 
  } } }
  for(int i = 0; i < nnz2; i+=4) {
    for(int k=i+1; k<i+4; ++k) {
      neuron_i_ind[k] = neuron_i_ind[k-1];  
  } }
}


// std::pair<int,int> convolution_layer_blocked((VTYPE (&synapse_val)[nnz1][Nn],VTYPE (&synapse_ind)[nnz1][Nn],
void convolution_layer_blocked(VTYPE (&synapse_val)[int(nnz1)][Nn], VTYPE (&synapse_ind)[int(nnz1)][Nn],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor], VTYPE (&synapse_ptr)[Ni],
                                   VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  // int c1=0,c2=0;
  // VTYPE sum[Nn]={0};
  int size_neuron_tile = 0;
  int size_synapse = 0;

  begin_roi();
  SB_CONFIG(test_config,test_size);
  // Tn is based on the size of scratchpad
  for (int n = 0; n < Nn; n += Tn) { 
    SB_CONST_SCR(0, 0, NYPAD*NXPAD*Tn/tile_factor);
    SB_WAIT_SCR_WR();
    printf("SCRATCHPAD RESET DONE!\n");
    // at the end, Tn output feature maps will be available
    for(int tile_no=0; tile_no<tile_factor; ++tile_no) { // tiling in neurons (TODO: check factor 8)
      for (int feature_map_id = 0; feature_map_id < Ni; ++feature_map_id) {

        // TODO: don't want these select statements
        size_synapse = synapse_ptr[feature_map_id+1] - synapse_ptr[feature_map_id];
        size_neuron_tile = neuron_ptr[(feature_map_id*tile_factor + tile_no)+1] - neuron_ptr[(feature_map_id*tile_factor + tile_no)];
        size_synapse = size_synapse > 0 ? size_synapse : nnz1;
        size_neuron_tile = size_neuron_tile > 0 ? size_neuron_tile: nnz1;
        size_synapse = 4;
        size_neuron_tile = 16;
        cout << "size_synapse: " << size_synapse << " size_neuron_tile: " << size_neuron_tile << "\n";
        
        int num_comp_inst = (size_neuron_tile/4)*(size_synapse/4)*4*4;
        // SB_FILL_MODE(STRIDE_DISCARD_FILL);
        for (int nn = n; nn < n+Tn; nn++) { 
          int nval_st = neuron_ptr[(feature_map_id*tile_factor + tile_no)];

          for(int is=0; is<(size_synapse/4)*4; ++is) {
            // read neuron
            SB_DMA_READ(&neuron_i_val[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_test_nval);
            SB_DMA_READ(&neuron_i_ind[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_test_nind);
          }

          // broadcast
          // read synapse: 9 reads
          SB_REPEAT_PORT((size_neuron_tile/4)*4);
          SB_DMA_READ(&synapse_val[synapse_ptr[feature_map_id]][nn], sizeof(VTYPE)*Nn, 4*sizeof(VTYPE), size_synapse/4, P_test_sval);
          SB_REPEAT_PORT((size_neuron_tile/4)*4);
          SB_DMA_READ(&synapse_ind[synapse_ptr[feature_map_id]][nn], sizeof(VTYPE)*Nn, 4*sizeof(VTYPE), size_synapse/4, P_test_sind);

          SB_CONST(P_test_Kx, Kxsim, num_comp_inst/4);
          SB_CONST(P_test_Nx, Nxsim, num_comp_inst/4);

          SB_CONST(P_test_init_addr, 0, 1); // In-0 = 0 (1 64-bit value)

          SB_RECURRENCE(P_test_rec, P_test_init_addr, (num_comp_inst-1)/4);

          // write partial sums to SCR
          SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
          SB_ATOMIC_SCR_OP(P_test_C, P_test_D, 0, num_comp_inst, 0);
          SB_WAIT_SCR_WR();
          SB_GARBAGE_SIMP(P_test_rec,1);
        }
      }
      // ReLU
      // Tn output feature maps in  dense format: should write in sparse after ReLU
      // SB_SCRATCH_DMA_STORE(0, sizeof(VTYPE)*Nn, 4*sizeof(VTYPE), NYPAD*NXPAD*Tn/tile_factor, &neuron_n[0][0][n]);
      SB_WAIT_ALL();
      printf("1 tile done\n");
    }
  }
  end_roi();
  sb_stats();
}



int main() {

  // VTYPE (*synapse_val)[Nn];
  // VTYPE (*synapse_ind)[Nn];
  // VTYPE (*synapse_ptr);
  // VTYPE (*neuron_i_val);
  // VTYPE (*neuron_i_ind);
  // VTYPE (*neuron_ptr);
  // VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];
  VTYPE (*synapse_val)[int(nnz1)][Nn];
  VTYPE (*synapse_ind)[int(nnz1)][Nn];
  VTYPE (*synapse_ptr)[Ni];
  VTYPE (*neuron_i_val)[int(nnz2)];
  VTYPE (*neuron_i_ind)[int(nnz2)];
  VTYPE (*neuron_ptr)[Ni*tile_factor];
  VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];


  cout << "allocating memory\n";

  synapse_val   = (VTYPE (*)[int(nnz1)][Nn])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ind   = (VTYPE (*)[int(nnz1)][Nn])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ptr        = (VTYPE (*)[Ni])  malloc(Ni*sizeof(VTYPE));
  neuron_i_val  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_i_ind  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_ptr        = (VTYPE (*)[Ni*tile_factor])  malloc(Ni*tile_factor*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_data(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr);
  modify_encoding(*synapse_ind,*neuron_i_ind);

  cout << "starting computation\n";

  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *neuron_n);
  end_roi();


  cout << "blocked computation complete!\n";  

  // compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
  return 0;
}
