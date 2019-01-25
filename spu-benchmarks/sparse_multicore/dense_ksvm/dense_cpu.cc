#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
using namespace std;

#define VTYPE float
#define C 0.1
#define tol 0.02
// #define max_passes 100
#define max_passes 1
//#define M 100 // number of instances
//#define N 10 // number of features
#define sigma 0.5
//#define ratio 0.6

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


// Dense data structures
uint64_t y[M];
uint32_t data[M][N];

double alpha[M];
double E[M];

float min(float a, float b){
    return a<b?a:b;
}
float max(float a, float b){
    return a>b?a:b;
}

float gauss_kernel(float x){
    return exp(-(x*x)/(2*sigma*sigma));
}

void eta_calc(long tid, int i, int j, double &dp, double &norm1, double &norm2){
  
  for(int k=0; k<N; ++k) {
    dp += data[i][k]*data[j][k];
    norm1 += data[i][k]*data[i][k];
    norm2 += data[j][k]*data[j][k];
  }

  // cout << "Eta calc done\n";
}

// dense only
void calc_duality_gap(long tid, double b, double &duality_gap){
  for(int i=0; i<M; ++i) {
    duality_gap += alpha[i]*y[i]*E[i];
  }
  duality_gap += b;
  // cout << "Duality calc done\n";
}

void kernel_err_update(long tid, int i, int j, double diff1, double diff2, double y1, double y2){
  uint32_t s1=0, s2=0;
 
  for(int m=0; m<M; ++m) { // m=M'
    for(int k=0; k<N; ++k) {
      s1 += data[i][k]*data[m][k];
      s2 += data[j][k]*data[m][k];
    }
  }
  // something something -- can just change the sparse part of the dfg

 // cout << "Kernel err calc done\n";
}

void train(long tid) {
  double b1, b2, b=0; // initial bias?

  double L = 0, H = 0;
  int passes=0;
  double old_alpha_i=0, old_alpha_j=0;
  double eta = 0;
  // float diff = 0;
  double diff = 0;
  int j=1, i=0;
  double duality_gap=0;
  double dual=0;

  while (passes<max_passes) {
    passes++;

    // Select new i and j such that E[i] is max and E[j] is min do in CGRA
    for(int k=0; k<M; ++k){
      if(E[k]>E[i])
        i=k;
      if(E[k]<E[j])
        j=k;
    }
	// std::cout << "i: " << i << " j: " << j << std::endl;

    // Step 1:
    old_alpha_i=alpha[i];
    old_alpha_j=alpha[j];
    if(y[i] != y[j]){
      L = max(0,alpha[j]-alpha[i]);
      H = min(C, C+alpha[j]-alpha[i]);
    } else {
      L = max(0, alpha[i]+alpha[j]-C);
      H = min(C, alpha[i]+alpha[j]);
    }
    // cout << "L=H?\n";
	// cout << L << " " << H << endl;
    if(L==H) continue;
    double inter_prod = 0, norm1 = 0, norm2 = 0;
	// cout << "Sent for eta calculation\n";
    SB_CONFIG(eta_config, eta_size);
    eta_calc(tid, i, j, inter_prod, norm1, norm2);
    eta = 2*inter_prod - norm1 - norm2;
    if(eta == 0) eta=2;
    // cout << "Eta was less\n";
    // if(eta >= 0) continue;
    // double diff2 = (y[j]*(E[i]-E[j]))/eta;
    diff = (y[j]*(E[i]-E[j]))/eta;
    alpha[j] = alpha[j] - diff;
    // alpha[j] = alpha[j] - diff2;
    // alpha[j] = alpha[j] - (y[j]*(E[i]-E[j]))/eta; // y should be stored in dense format?
    if(alpha[j]>H){
        alpha[j]=H;
    } else if(alpha[j]<L){
        alpha[j]=L;
    }
    // diff = alpha[j]-old_alpha_j;
    /*
    cout << "Diff was less\n";
    if(diff < 1e-5) {
        continue;
    }
    */
    // double diff1 = (y[i]*y[j])*(diff2);
    double diff1 = (y[i]*y[j])*(diff);
    // alpha[i]=alpha[i]+ (y[i]*y[j])*(old_alpha_j-alpha[j]);
    alpha[i]=alpha[i]+ diff1;

    // b1 = b - E[i] - y[i]*diff;
    // b2 = b - E[j] - y[i]*diff;

    b1 = b - E[i] - y[i]*diff*norm1 - y[j]*diff*inter_prod;
    b2 = b - E[j] - y[i]*diff*inter_prod - y[j]*diff*norm2;

    if(alpha[i]>0 && alpha[i]<C){
        b = b1;
    } else if(alpha[j]>0 && alpha[j]<C){
        b = b2;
    } else {
        b = (b1+b2)/2;
    }
    dual = dual - diff/y[i]*(E[i]-E[j]) + eta/2*(diff/y[i])*(diff/y[i]);

	// cout << "Sent for kernel err calculation\n";
    kernel_err_update(tid, i, j, diff1, diff, y[i], y[j]);

    duality_gap = 0;
	// cout << "Sent for duality gap calculation\n";
    calc_duality_gap(tid, b, duality_gap);
    /*
    for(int k=0; k<M; ++k){
      duality_gap += alpha[k]*y[k]*E[k];
    }
    duality_gap += b;
    */

  }
}




int main(){

  int nnz1=0, nrows1=0;
  int *row_ptr;
  // std::pair<VTYPE,VTYPE> *matrix1;
  float *data_val;
  float *data_ind;

  double *y;
  y = (double*)malloc(M*sizeof(double));
  data_val = (float*)malloc(M*N*ratio*sizeof(float));
  data_ind = (float*)malloc(M*N*ratio*sizeof(float));
  row_ptr = (int*)malloc((M+1)*sizeof(int));


  float *temp_val;
  float *temp_ind;
  double *out;
  out = (double*)malloc(1*sizeof(double));
  temp_val = (float*)malloc(N*ratio*sizeof(float));
  temp_ind = (float*)malloc(N*ratio*sizeof(float));

  int id=0;


  FILE *m1_file;
  char lineToRead[5000];

  m1_file = fopen("input.data", "r");
  printf("Start reading matrix1\n");

  while(fgets(lineToRead, 500000, m1_file) != NULL) {
    sscanf(lineToRead, "%lf %f:%f %f:%f %f:%f %f:%f %f:%f %f:%f", &out[0], &temp_ind[0], &temp_val[0], &temp_ind[1], &temp_val[1], &temp_ind[2], &temp_val[2], &temp_ind[3], &temp_val[3], &temp_ind[4], &temp_val[4], &temp_ind[5], &temp_val[6]);
    // y[id] = (int32_t)(out[0] * (1<<FxPnt));
    y[id] = out[0];
    for(int j=0; j<N*ratio; ++j){
      // data_val[(int)(id*N*ratio + j)] = (int32_t)(temp_val[j] * (1<<FxPnt));
      // data_ind[(int)(id*N*ratio + j)] = (int32_t)(temp_ind[j] * (1<<FxPnt));

      data_val[(int)(id*N*ratio + j)] = temp_val[j];
      data_ind[(int)(id*N*ratio + j)] = temp_ind[j];
    }
    row_ptr[id] = id*N*ratio;
    id++;
  }
  row_ptr[M] = N*M*ratio;

  printf("Finished reading input data\n");

  train(data_val, data_ind, row_ptr, y);
  begin_roi();
  train(data_val, data_ind, row_ptr, y);
  end_roi();
  printf("svm training done\n");
 
  return 0;
}


/*
int main(){

  int nnz1=0, nrows1=0;
  int *row_ptr1;
  std::pair<VTYPE,VTYPE> *matrix1;

  FILE *m1_file;
  char lineToRead[5000];

   m1_file = fopen("datasets/ddup_pdb1HYS.mtx", "r");

   int id = -1; // pointer to location in values
   int row_id = 0;
   int prev_row_id = -1;
   VTYPE t1 = 0, t2 = 0;
   bool start=false;
   // int x = 0, y = 0;
   printf("Start reading matrix1\n");
   while(fgets(lineToRead, 500000, m1_file) != NULL) {

       if(*lineToRead == '%')
           continue;
       if(!start){
           sscanf(lineToRead, "%d %d %d", &nrows1, &nrows1, &nnz1);
           row_ptr1 = (int*)malloc((nrows1+1)*sizeof(int));
           for(int i=0; i<nrows1; ++i){
               row_ptr1[i] = -1;
           }
           matrix1 = (std::pair<VTYPE,VTYPE>*)malloc(nnz1*sizeof(std::pair<VTYPE,VTYPE>));
           start = true;
       }

       else {
            ++id;
            sscanf(lineToRead, "%f %d %f", &t1, &row_id, &t2);
            row_id--;
            // matrix1[id] = std::make_pair((t1-1)%N, t2%N);
            matrix1[id] = std::make_pair((t1-1), t2);
            if(row_ptr1[row_id]==-1) {
                row_ptr1[row_id] = id;
                if(prev_row_id!=-1){
                  for(int i=prev_row_id+1; i<row_id; ++i){
                      // assert(row_ptr1[i]==-1 && "some problem\n");
                      row_ptr1[i] = id;
                  }
                }
                prev_row_id = row_id;
            }
       }
    }
    row_ptr1[nrows1] = nnz1-1;

    VTYPE *y;
    srand(1); // this is seed is to be used by the algorithm
    y = (VTYPE*)malloc(M*sizeof(VTYPE));
    for(int i=0; i<M; ++i){
        y[i] = rand()%10;
    }

   printf("Finished reading input data\n");

  train(matrix1, row_ptr1, nnz1, y);
  printf("svm training done\n");
  
  return 0;
}
*/
