#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <inttypes.h>
#include <math.h>
#include <sstream>
#include <sys/time.h>
#include <vector>
using namespace std;

#define VTYPE float
#define C 0.1
#define tol 0.02
#define max_passes 1
#define sigma 0.5

// input train set
double y[M];
vector<float> data_val;
vector<float> data_ind;
int data_ptr[M+1]; // save the accumulated indices
 
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


float min(float a, float b){
    return a<b?a:b;
}
float max(float a, float b){
    return a>b?a:b;
}

double dot_prod_sparse(int ptr1, int end1, int ptr2, int end2){
  double accum = 0;
  while(ptr1 <= end1 && ptr2 <= end2){
    if(data_ind[ptr1] == data_ind[ptr2]){
        accum += data_val[ptr1]*data_val[ptr2];
        ptr1++; ptr2++;
        }
        else{
          if(data_ind[ptr1] <= data_ind[ptr2])
            ptr1++;
          else
            ptr2++;
        }
      }
    return accum;
}

double norm(int start_ptr, int end_ptr){
    VTYPE output=0;
    for(int i=start_ptr; i<end_ptr; ++i){
        output += data_val[i]*data_val[i];
    }
    return output;
}

float gauss_kernel(float x){
    return exp(-(x*x)/(2*sigma*sigma));
}

double kernel_func(float *alpha, VTYPE *y, int i, float b, int m){
  VTYPE output = 0;
  VTYPE temp;
  // jst run this 2 times
  for(int j=0; j<m; ++j){
      temp = dot_prod_sparse(data_ptr[i], data_ptr[i+1], data_ptr[j], data_ptr[j+1]);
      temp = gauss_kernel(temp); // apply the gaussian kernel
      temp *= alpha[j]*y[j];
      output += temp;
  }

  return output+b-y[i];
}


// void train(std::pair<VTYPE,VTYPE> *data, int *data_ptr, int nnz, VTYPE *y){
void train(){
    float alpha[M];
    float b1, b2, b=0; // initial bias?
    for(int i=0; i<M; ++i){
        alpha[i]=0.1;
    }
    double E[M]; // let's see the dim of E
     
    for(int k=0; k<M; ++k){
      E[k] = -y[k];
    }


    float L = 0, H = 0;
    int passes=0;
    // int num_changed_alphas=0;
    float old_alpha_i=0, old_alpha_j=0;
    VTYPE eta = 0;
    float diff = 0;
    int i=0, j=0;

    // int Iup = 0;
    // int Ilow = 0;
    double duality_gap = 0;
    double dual = 0;

    // while (duality_gap <= tol*dual && passes<max_passes) {
    // while (duality_gap <= tol*dual || passes<max_passes) {
    while (passes<max_passes) {
     passes++;

     // cout << "Pass number: " << passes << "\n";
     // Select new i and j such that E[i] is max and E[j] is min
     for(int k=0; k<M; ++k){
       if(E[k]>E[i])
         i=k;
       if(E[k]<E[j])
         j=k;
     }

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
     if(L==H) continue;
     VTYPE inter_prod = dot_prod_sparse(data_ptr[i], data_ptr[i+1], data_ptr[j], data_ptr[j+1]);
     VTYPE intra_prod1 = norm(data_ptr[i], data_ptr[i+1]);
     VTYPE intra_prod2 = norm(data_ptr[j], data_ptr[j+1]);
     eta = 2*inter_prod - intra_prod1 - intra_prod2;
    
     if(eta >= 0) eta=2;

     diff = (y[j]*(E[i]-E[j]))/eta;
     alpha[j] = alpha[j] - diff;

     if(alpha[j]>H){
         alpha[j]=H;
     } else if(alpha[j]<L){
         alpha[j]=L;
     }


     double diff1 = (y[i]*y[j])*(diff);
     alpha[i]=alpha[i]+ diff1;

     // alpha[i]=alpha[i]+ (y[i]*y[j])*(old_alpha_j-alpha[j]);
     b1 = b - E[i] - y[i]*diff*intra_prod1 - y[j]*diff*inter_prod;
     b2 = b - E[j] - y[i]*diff*inter_prod - y[j]*diff*intra_prod2;
     if(alpha[i]>0 && alpha[i]<C){
         b = b1;
     } else if(alpha[j]>0 && alpha[j]<C){
         b = b2;
     } else {
         b = (b1+b2)/2;
     }
     
     dual = dual - diff/y[i]*(E[i]-E[j]) + eta/2*(diff/y[i])*(diff/y[i]);

     // Step 2: (M is number of instances)
     // calculate all the new parameters
     // cout << "Came to calculate new E[i]\n";
    
     // kernel error update
     for(int k=0; k<M; ++k){
       E[k] = E[k] + (diff1)*y[i]*gauss_kernel(dot_prod_sparse(data_ptr[k], data_ptr[k+1],data_ptr[i], data_ptr[i+1])) + (diff)*y[j]*gauss_kernel(dot_prod_sparse(data_ptr[k], data_ptr[k+1],data_ptr[j], data_ptr[j+1]));
     }

     duality_gap = 0;
     for(int k=0; k<M; ++k){
       duality_gap += alpha[k]*y[k]*E[k];
     }
     duality_gap += b;
    }

    // end_roi();
    // cout << "Algorithm converged at number of passes: " << passes << endl;
}


int main(){


  FILE *m1_file;
  char lineToRead[5000];
  string str(file);

  m1_file = fopen(str.c_str(), "r");

  data_ptr[0]=0;
  int inst_id=0;
  while(fgets(lineToRead, 5000, m1_file) != NULL) {
	std::string raw(lineToRead);
	std::istringstream iss(raw.c_str());
	char ignore;
	float x;
	int ind;

	iss >> x;
	y[inst_id] = x; // DOUBLE_TO_FIX(x);

	while(iss >> ind) {
	  iss >> ignore >> x;
	  data_ind.push_back(ind);
	  data_val.push_back(x); // DOUBLE_TO_FIX(x));
	}

    inst_id++;
    data_ptr[inst_id] = data_ind.size(); // data_ptr[inst_id-1] + data_val[inst_id-1].size();
  }

  printf("Finished reading input data\n");

  train();
  begin_roi();
  train();
  end_roi();
  printf("svm training done\n");
 
  return 0;
}
