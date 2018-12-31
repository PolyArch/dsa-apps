#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

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
 
static uint64_t ticks;

static __inline__ uint64_t rdtsc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static void begin_roi() {
  ticks=rdtsc();
}

static void end_roi()   {
  ticks=(rdtsc()-ticks);
  printf("ticks: %lu\n", ticks);
}


// working on act of neuron_i_val[z][y]
// working on weight of synapse_val[y][x]
void kernel(int x, int y, int z) {

  unsigned size_neuron_tile = neuron_i_val[z][y].size();
  unsigned size_synapse = synapse_val[y][x].size();

  // cout << size_synapse << " " << synapse_ind[y][x].size() << endl;
  // cout << size_neuron_tile << " " << neuron_i_ind[z][y].size() << endl;

  // int num_comp_inst = size_synapse*size_neuron_tile;
  int out_ind=0; int nx, ny, sx, sy;
  int cur_wgt_ind=0; int cur_act_ind=0;

  for(unsigned w=0; w<size_synapse; ++w) {
    // cout << "w: " << w << endl;
    cur_wgt_ind += synapse_ind[y][x][w];
    sx = cur_wgt_ind%(Kx*Ky); sy = cur_wgt_ind/(Kx*Ky);
    for(unsigned n=0; n<size_neuron_tile; ++n) {
      cur_act_ind += neuron_i_ind[z][y][n];
      nx = cur_act_ind%Nx; ny = cur_act_ind/Nx;
      out_ind = (nx-sx-1)*(Tx-Kx-1)+(ny-sy-1);
      // cout << "out_ind: " << out_ind << endl;
      out_ind = out_ind%((Nx-Kx+1)*(Ny-Ky+1));
      // cout << "out index: " << out_ind << endl;
      // FIXME: check this x
      out_n[x][out_ind] += synapse_val[y][x][w]*neuron_i_val[z][y][n];
    }
    // cout << "1 iteration done!" << endl;
    cur_act_ind=0;
  }

  // RELU here

  // cout << "MULT: " << (size_neuron_tile*size_synapse) << endl;

}

void convolution_layer_blocked(long tid) {
  // int n_count = halo_count(tid);
  int stride = (Nx*Ny)/(Tx*Ty);
  /*for(int i=0; i<Nn/Tn; ++i)*/ int i=0; {
	/*for(int j=0; j<Ni; ++j) */ int j=0; {
      if(tid==i*Ni+j) {
        // TODO: see if we need to do this!
        // load_weights_in_linear_scratch(j,i);
        // broadcast_weights(tid, j, i);
        // count++;
      }
      kernel(i,j,tid);
      begin_roi();
      kernel(i,j,tid);
      end_roi();
      // sb_stats();
	  // all of them use the same weights
      /*
	  for(int k=tid*stride; k<stride*(1+tid); ++k) {
		kernel(i,j,k);
	  }
      */
      // send_halos(tid);
      // SB_WAIT_DF(n_count, 0);
	}
  }
}


void read_weights() {

  char lineToRead[5000];
  string str(net_name);

  char x2[100] = "datasets/";
  char y2[100] = "/wgt_index.data";
  FILE *weight_ind_file = fopen(strcat(strcat(x2,str.c_str()),y2), "r");
 
  while(fgets(lineToRead, 5000, weight_ind_file)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    uint16_t var;

	/*for(int i=0; i<Ni; ++i)*/ int i=0;{
	  /* for(int j=0; j<Nn/Tn; ++j)*/ int j=0;{
		iss >> var;
		synapse_ind[i][j].push_back(var);
		synapse_val[i][j].push_back(3);
	  }
	}
	
  }
  fclose(weight_ind_file);




/*
  char x1[100] = "datasets/";
  char y1[100] = "/wgt_val.data";
  
  FILE *weight_val_file = fopen(strcat(strcat(x1,str.c_str()),y1), "r");
  
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
  */

  char x3[100] = "datasets/";
  char y3[100] = "/wgt_ptr.data";
  FILE *weight_ptr_file = fopen(strcat(strcat(x3,str.c_str()),y3), "r");
 
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

  char lineToRead[5000];
  string str(net_name);

  char x2[100] = "datasets/";
  char y2[100] = "/act_index.data";
  FILE *act_ind = fopen(strcat(strcat(x2,str.c_str()),y2), "r");


  // FILE *act_ind = fopen("datasets/act_index.data","r"); 
  
  while(fgets(lineToRead, 5000, act_ind)!=NULL){
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
    uint16_t var;

	/*for(int i=0; i<dim1; ++i)*/ int i=0;{
	  /*for(int j=0; j<Ni; ++j)*/ int j=0;{
		iss >> var;
        // cout << "Weights: " << var << endl;
		neuron_i_ind[i][j].push_back(var);
		neuron_i_val[i][j].push_back(3);
	  }
	}
  }
  fclose(act_ind);


/*
  char x1[100] = "datasets/";
  char y1[100] = "/act_val.data";
  FILE *act_val = fopen(strcat(strcat(x1,str.c_str()),y1), "r");


  // FILE *act_val = fopen("datasets/act_val.data","r"); 

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
  */

  char x3[100] = "datasets/";
  char y3[100] = "/act_ptr.data";
  FILE *act_ptr = fopen(strcat(strcat(x3,str.c_str()),y3), "r");
  
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
  
  // Initialize pointers
  for(int i=0; i<(Nn*Ni/Tn); ++i) {
    synapse_ptr[i]=0;
  }

  for(int i=0; i<(Nx*Ny*Ni)/(Tx*Ty); ++i) {
    neuron_i_ptr[i]=0;
  }

  read_weights();
  cout << "Done reading weights\n";

  read_activations();
  cout << "Done reading activations\n";

  printf("Done reading file!\n");

}


int main() {

  cout << "initializing arrays\n";

  fill_convolution_data();

  cout << "starting computation\n";

  convolution_layer_blocked(0);


  cout << "blocked computation complete!\n";  

  // compare((uint16_t*)*neuron_n,(uint16_t*)*neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
  return 0;
}

