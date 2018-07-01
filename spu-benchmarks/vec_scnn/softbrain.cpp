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

// #define Ni 64
// #define tile_factor 8
// #define Ni 64
#define tile_factor 8
// #define tile_factor 1
// #define Tn 1 // for now


#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

// #define NYPAD (Ny)
// #define NXPAD (Nx)

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
#define nnz_ne (neuron_sp*Tx*Tx)
#define nnz2 (25076*Ni)


void fill_convolution_data(VTYPE (&synapse_val)[Ni][int(nnz1)],VTYPE (&synapse_ind)[Ni][int(nnz1)],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor+1], VTYPE (&synapse_ptr)[Nn][Ni+1]) {

  char lineToRead[1000];
  int id=0;
  // FILE* neuron_file = fopen("dataset_rle.txt","r");
  FILE* neuron_file = fopen("input_neuron.data","r");
  while(fgets(lineToRead, 5000, neuron_file) != NULL) {
    sscanf(lineToRead, "%hu %hu %hu %hu %hu %hu %hu %hu", &neuron_i_val[id], &neuron_i_ind[id], &neuron_i_val[id+1], &neuron_i_ind[id+1], &neuron_i_val[id+2], &neuron_i_ind[id+3], &neuron_i_val[id+3], &neuron_i_ind[id+3]);
    id+=4;
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
  while(fgets(lineToRead, 5000, synapse_file) != NULL) {
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
  
  /*
  for(int j = 0; j < Nn; ++j) {
    for(int i = 0; i < nnz1; ++i) {
      synapse_val[j][i] = static_cast <VTYPE> (rand()%500); // / static_cast <VTYPE> (500);
  } }
  for(int j = 0; j < Nn; ++j) {
    for(int i = 0; i < nnz1; i+=4) {
      synapse_ind[j][i] = 3; 
      synapse_ind[j][i+1] = 0; 
      synapse_ind[j][i+2] = 1; 
      synapse_ind[j][i+3] = 1; 

  } }
*/
  /*
   // for synapse
  for(int j=0; j<Nn; ++j) {
    for(int i = 0; i < Ni; ++i) {
      // synapse_ptr[j][i] = 4*(j*Ni+i);
      synapse_ptr[j][i] = 4*i; // it should be reset at each Nn i think
    }
    // synapse_ptr[j][Ni] = nnz1;
    synapse_ptr[j][Ni] = nnz1*(j+1);
  }
  */

}

void modify_encoding(VTYPE (&synapse_ind)[Nn][int(nnz1)],VTYPE (&neuron_i_ind)[int(nnz2)]) {
  for(int j = 0; j < Nn; ++j) {
    for(int i = 0; i < nnz1; i+=4) {
      for(int k=i+1; k<i+4; ++k) {
        // synapse_ind[k][j] += synapse_ind[k-1][j]; 
        synapse_ind[j][k] += synapse_ind[j][k-1]; 
  } } }
  for(int i = 0; i < nnz2; i+=4) {
    for(int k=i+1; k<i+4; ++k) {
      neuron_i_ind[k] += neuron_i_ind[k-1];  
  } }
}


// std::pair<int,int> convolution_layer_blocked((VTYPE (&synapse_val)[nnz1][Nn],VTYPE (&synapse_ind)[nnz1][Nn],
void convolution_layer_blocked(VTYPE (&synapse_val)[Ni][int(nnz1)], VTYPE (&synapse_ind)[Ni][int(nnz1)],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor+1], VTYPE (&synapse_ptr)[Nn][Ni+1],
                                   VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  // int c1=0,c2=0;
  // VTYPE sum[Nn]={0};
  int size_neuron_tile = 0;
  int size_synapse = 0;
  int num_comp_inst = 0;


  VTYPE sy_ind1, sy_ind2, sy_ind3, sy_ind4;
  VTYPE offset1, offset2, offset3, offset4;
  uint64_t weightsim;

  begin_roi();
  SB_CONFIG(test_config,test_size);
  // Tn is based on the size of scratchpad: NOT NEEDED FOR 1 PE
  // for (int n = 0; n < Nn; n += Tn) { 
    // at the end, Tn output feature maps will be available
    for(int tile_no=0; tile_no<tile_factor; ++tile_no) { // tiling in neurons (TODO: decide according to new numbers)
      SB_CONST_SCR(0, 0, NYPAD*NXPAD*Tn/tile_factor);
      SB_WAIT_SCR_WR();
      SB_WAIT_SCR_RD();
      // Tx*Ty*Tn output feature maps = here Tx*Ty (Nx*Ny/tile_factor)
      for (int feature_map_id = 0; feature_map_id < Ni; ++feature_map_id) {
          // load feature_map_idth neuron tile into scratchpad

        size_synapse = nnz_syn;
        size_neuron_tile = nnz_ne; // size of 1 tile

        SB_REPEAT_PORT(size_neuron_tile/4);
        SB_DMA_READ(&synapse_val[feature_map_id][0], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_synapse/4, P_test_sval);

        int nval_st = neuron_ptr[(feature_map_id*tile_factor + tile_no)];
        for(int is=0; is<size_synapse/4; ++is) {
            // read neuron
            SB_DMA_READ(&neuron_i_val[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_test_nval);
            SB_DMA_READ(&neuron_i_ind[nval_st], 4*sizeof(VTYPE), 4*sizeof(VTYPE), size_neuron_tile/4, P_test_nind);
          }
        // for (int nn = n; nn < n+Tn; nn++) { // do we need this loop: instead we should loop over Ni i think 
        for (int nn = 0; nn < Tn; nn++) {
        // for (int nn = 0; nn < 1; nn++) {
          
          // size_synapse = synapse_ptr[feature_map_id][feature_map_id+1] - synapse_ptr[nn][feature_map_id];
          // size_synapse = 4;
          size_synapse = (size_synapse/4)*4;
          size_neuron_tile = (size_neuron_tile/4)*4;
          num_comp_inst = (size_synapse*size_neuron_tile)/(4*4);

          // this should renew at each new synapse: DO SOMETHING FOR IT
          for(int is=0; is<size_synapse/Tn; is+=4) {
            // this should run for size_neuron_tile/4 instances
            sy_ind1 = synapse_ind[feature_map_id][is];
            sy_ind2 = synapse_ind[feature_map_id][is+1];
            sy_ind3 = synapse_ind[feature_map_id][is+2];
            sy_ind4 = synapse_ind[feature_map_id][is+3];
            // I think this init has to be taken care of by padding?
            offset1 = Kx*Nx - (sy_ind1/Kx + Nx*(sy_ind1%Kx)) + nn*Nx*Kx;
            offset2 = Kx*Nx - (sy_ind2/Kx + Nx*(sy_ind2%Kx)) + nn*Nx*Kx;
            offset3 = Kx*Nx - (sy_ind3/Kx + Nx*(sy_ind3%Kx)) + nn*Nx*Kx;
            offset4 = Kx*Nx - (sy_ind4/Kx + Nx*(sy_ind4%Kx)) + nn*Nx*Kx;
            // printf("start fm index is: %d\n",sval_st);
            // printf("print different indices into the index: %d %d %d %d\n", sy_ind1, sy_ind2, sy_ind3, sy_ind4);
            // printf("print different weight offsets into the index: %d %d %d %d\n", offset1, offset2, offset3, offset4);
            weightsim = (offset1 | offset2 << 16 | (offset3 & 0xFFFFFFFFFFFFFFFF) << 32 | (offset4 & 0xFFFFFFFFFFFFFFFF) << 48);
            
            // SB_CONST(P_test_init_addr, Nxsim, 1); // In-0 = 0 (1 64-bit value)
            SB_CONST(P_test_init_addr, weightsim, 1); // In-0 = 0 (1 64-bit value)
            SB_RECURRENCE(P_test_rec, P_test_init_addr, size_neuron_tile/4-1);
            SB_GARBAGE_SIMP(P_test_rec,1);
            // write partial sums to SCR
            SB_CONFIG_ATOMIC_SCR_OP(T16, T16, T16);
            // SB_ATOMIC_SCR_OP(P_test_C, P_test_D, 0, size_neuron_tile*4, 0);
            SB_ATOMIC_SCR_OP(P_test_C, P_test_D, 0, size_neuron_tile, 0);
            SB_WAIT_SCR_WR();
            SB_WAIT_SCR_RD();

          }

        }
        SB_WAIT_ALL();
        // printf("done for 1 feature id: %d\n",feature_map_id);
      }
      // ReLU
      // Tn output feature maps in  dense format: should write in sparse after ReLU
      // SB_SCRATCH_DMA_STORE(0, sizeof(VTYPE)*Nn, 4*sizeof(VTYPE), NYPAD*NXPAD*Tn/tile_factor, &neuron_n[0][0][n]);
      SB_WAIT_ALL();
      // printf("1 tile done\n");
    }
  // }
  end_roi();
  sb_stats();
}



int main() {

  VTYPE (*synapse_val)[Ni][int(nnz1)];
  VTYPE (*synapse_ind)[Ni][int(nnz1)];
  VTYPE (*synapse_ptr)[Nn][Ni+1];
  VTYPE (*neuron_i_val)[int(nnz2)];
  VTYPE (*neuron_i_ind)[int(nnz2)];
  VTYPE (*neuron_ptr)[Ni*tile_factor+1];
  VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];


  cout << "allocating memory\n";

  synapse_val   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ind   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ptr        = (VTYPE (*)[Nn][Ni+1])  malloc(Nn*(Ni+1)*sizeof(VTYPE));
  neuron_i_val  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_i_ind  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_ptr        = (VTYPE (*)[Ni*tile_factor+1])  malloc((Ni+1)*tile_factor*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_data(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr);
  // modify_encoding(*synapse_ind,*neuron_i_ind);

  cout << "starting computation\n";

  //Blocked Version
  // begin_roi();
  convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *synapse_ptr, *neuron_n);
  // end_roi();


  cout << "blocked computation complete!\n";  

  // compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
  return 0;
}
