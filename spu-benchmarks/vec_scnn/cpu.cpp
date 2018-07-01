#include <iostream>
#include <stdlib.h>
#include <string>
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
#define Nn (Tn)

#define Kxsim (Kx | Kx << 16 | (Kx & 0xFFFFFFFFFFFFFFFF) << 32 | (Kx & 0xFFFFFFFFFFFFFFFF) << 48)
#define Nxsim (Nx | Nx << 16 | (Nx & 0xFFFFFFFFFFFFFFFF) << 32 | (Nx & 0xFFFFFFFFFFFFFFFF) << 48)

// #define Ni 64
// #define tile_factor 8
// #define Ni 64
#define tile_factor 8
// #define tile_factor 1
// #define Tn 1 // for now


// #define NYPAD (Ny+Ky)
// #define NXPAD (Nx+Kx)

#define NYPAD (Ny)
#define NXPAD (Nx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Ni)

#define nnz_syn (synapse_sp*Kx*Ky*Tn)
#define nnz1 (synapse_sp*Kx*Ky*Tn)
// #define nnz2 NYPAD*NXPAD*Ni*sparsity_n
#define nnz_ne (neuron_sp*Tx*Tx)
#define nnz2 (25076*Ni)

void fill_convolution_data(VTYPE (&synapse_val)[Ni][int(nnz1)],VTYPE (&synapse_ind)[Ni][int(nnz1)],
                                   VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                   VTYPE (&neuron_ptr)[Ni*tile_factor+1]) {

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
}

// std::pair<int,int> convolution_layer_blocked((VTYPE (&synapse_val)[nnz1][Nn],VTYPE (&synapse_ind)[nnz1][Nn],
void convolution_layer_blocked(VTYPE (&synapse_val)[Nn][int(nnz1)], VTYPE (&synapse_ind)[Nn][int(nnz1)],
                                               VTYPE (&neuron_i_val)[int(nnz2)], VTYPE (&neuron_i_ind)[int(nnz2)],
                                               VTYPE (&neuron_ptr)[Ni*tile_factor+1], VTYPE (&neuron_n)[NYSCL*NXSCL*Nn]) {

// int c1=0,c2=0;
  // VTYPE sum[Nn]={0};
  int size_neuron_tile = 0;
  int size_synapse = 0;
  int num_comp_inst = 0;
  VTYPE sy_ind, sy_val;
  VTYPE ny_ind, ny_val;
  VTYPE out_prod, out_ind, out_ind_prev = Nx;

  for(int i=0; i<NYSCL*NXSCL*Nn; ++i){
    neuron_n[i]=0;
  }

  begin_roi();
  // for (int n = 0; n < Nn; n += Tn) { 
    // at the end, Tn output feature maps will be available
    for(int tile_no=0; tile_no<tile_factor; ++tile_no) { // tiling in neurons (TODO: check factor 8)
      // Tx*Ty*Tn output feature maps = here Tx*Ty (Nx*Ny/tile_factor)
      for (int feature_map_id = 0; feature_map_id < Ni; ++feature_map_id) {
        // load feature_map_idth neuron tile into scratchpad

        // size_neuron_tile = neuron_ptr[(feature_map_id*tile_factor + tile_no)+1] - neuron_ptr[(feature_map_id*tile_factor + tile_no)];
      
        size_synapse = nnz_syn;
        size_neuron_tile = nnz_ne; // size of 1 tile
        int nval_st = neuron_ptr[(feature_map_id*tile_factor + tile_no)];

        for (int nn = 0; nn < Tn; nn++) { 
          
          size_synapse = (size_synapse/4)*4;
          size_neuron_tile = (size_neuron_tile/4)*4;
          num_comp_inst = (size_synapse*size_neuron_tile)/4;

          for(int weight=0; weight<size_synapse/Tn; ++weight) {
            sy_ind = synapse_ind[feature_map_id][nn+weight];
            sy_val = synapse_val[feature_map_id][nn+weight];
            out_ind_prev = Nx - (sy_ind/Kx + Nx*(sy_ind%Kx));
            for(int is=0; is<size_neuron_tile; ++is) {
              ny_ind = neuron_i_ind[nval_st+is];
              ny_val = neuron_i_val[nval_st+is];

              out_prod = sy_val*ny_val;
              out_ind = out_ind_prev + ny_ind;
              // printf("index into the output vector is:%d\n",out_ind);
              out_ind_prev = out_ind;

              neuron_n[out_ind] += out_prod;
            }
          }
        }
      }
      // temp += out_prod;
      // out_ind = out_ind > 0 ? out_ind : -out_ind;
      // printf("index into the output feature map: %d\n",out_ind);
      // printf("syind: %d nyind: %d out_ind_prec: %d\n", sy_ind, ny_ind, out_ind_prev);

    }
  // }
  end_roi();
}



int main() {

  VTYPE (*synapse_val)[Ni][int(nnz1)];
  VTYPE (*synapse_ind)[Ni][int(nnz1)];
  // VTYPE (*synapse_ptr)[Nn][Ni+1];
  VTYPE (*neuron_i_val)[int(nnz2)];
  VTYPE (*neuron_i_ind)[int(nnz2)];
  VTYPE (*neuron_ptr)[Ni*tile_factor+1];
  VTYPE (*neuron_n)[NYSCL*NXSCL*Nn];
  // VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];


  cout << "allocating memory\n";

  synapse_val   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  synapse_ind   = (VTYPE (*)[Ni][int(nnz1)])  malloc(int(nnz1)*Nn*sizeof(VTYPE));
  // synapse_ptr        = (VTYPE (*)[Nn][Ni+1])  malloc(Nn*(Ni+1)*sizeof(VTYPE));
  neuron_i_val  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_i_ind  = (VTYPE (*)[int(nnz2)])malloc(int(nnz2)*sizeof(VTYPE));
  neuron_ptr        = (VTYPE (*)[Ni*tile_factor+1])  malloc((Ni+1)*tile_factor*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[(NYSCL*NXSCL*Nn)])malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));
  // neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_data(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr);

  cout << "starting computation\n";

  //Blocked Version
  // begin_roi();
  convolution_layer_blocked(*synapse_val, *synapse_ind, *neuron_i_val, *neuron_i_ind, *neuron_ptr, *neuron_n);
  // end_roi();


  cout << "blocked computation complete!\n";  

  // compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
  return 0;
}
