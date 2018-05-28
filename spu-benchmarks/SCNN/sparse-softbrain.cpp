#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

std::pair<int,int> convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int c1=0,c2=0;
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) { // Ny
    int yout = yy/Sy;
    for (int xx = 0; xx < Nx; xx += Tx) { // Nx
        int xout = xx/Sx;
        for (int nn = 0; nn < Nn; nn += Tn) { // Nn
          SB_CONST_SCR(0, 0, Tn); // see the datatype size here
          // read synapse
          SB_DMA_READ(&synapse[0][0][nn][0], sizeof(VTYPE)*Ni, sizeof(VTYPE), Ky*Kx*Tn*Ni, P_test_A);
          // read neuron
          SB_DMA_READ(&neuron_i[yy][xx][0], sizeof(VTYPE)*Ni, sizeof(VTYPE), (Ty/Sy+Ky)*(Tx/Sx+Kx)*Ni, P_test_B);
          // write partial sums to SCR
          SB_ATOMIC_SCR_OP(P_test_C, P_test_D, 0, Ni*Ty*Tx*Ky*Kx*Tn, 0);
          SB_WAIT_SCR_WR();
          SB_SCRATCH_DMA_STORE(0, sizeof(VTYPE), sizeof(VTYPE), Tn, &sum[nn]);
          /*
          for (int i = 0; i < Ni; i++) { // Ni
            for (int y = yy; y < yy + Ty; y += Sy) { // Ny
              for (int x = xx; x < xx + Tx; x += Sx) { // Nx;
                for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                  for (int kx = 0; kx < Kx; kx++) {
                    for (int n = nn; n < nn + Tn; n++) { // Nn
                      VTYPE sv = synapse[ky][kx][n][i];
                      VTYPE nv = neuron_i[ky + y][kx + x][i];
                      sum[n] += sv*nv;
                    }
                  }
                }
              }
            }
          }
          */
          //transfer
          for (int n = nn; n < nn + Tn; n++) {
            neuron_n[yout][xout][n] = sum[n] > 0 ? sum[n] : sum[n]/4;
          }
       }
       xout++; 
     }
     yout++;
  }
}



int main(const int argc, const char** argv) {
  cout << "allocating memory\n";

  // synapse_val[nnz1][Nn], synapse_ind[nnz1][Nn] // Ky*Kx*Ni
  // neuron_i_val[nnz2], neuron_n_ind[nnz2] // Ny*Nx*Ni
  // neuron_n_val[nnz1], neuron_n_ind[nnz1]
  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_shared_simple(*synapse,*neuron_i);

  cout << "starting computation\n";

  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
  end_roi();


  cout << "blocked computation complete!\n";  

  compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
}
