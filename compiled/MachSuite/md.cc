/*
Implemenataion based on:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010.
*/

#include <cstdint>
#include "../Common/test.h"

#ifndef N
#define N 256
#endif

#ifndef M
#define M 16
#endif

#ifndef U
#define U 1
#endif

void md_kernel(double force_x[],
               double force_y[],
               double force_z[],
               double position_x[],
               double position_y[],
               double position_z[],
               int64_t NL[])
{

  #pragma ss config
  {
    double delx, dely, delz, r2inv;
    double r6inv, potential, force, j_x, j_y, j_z;
    double i_x, i_y, i_z, fx, fy, fz;
    int32_t i, j, jidx;

    #pragma ss stream
    for (i = 0; i < N; i++){
      i_x = position_x[i];
      i_y = position_y[i];
      i_z = position_z[i];
      fx = 0;
      fy = 0;
      fz = 0;

      #pragma ss dfg dedicated unroll(U)
      for(j = 0; j < M; j++){
        // Get neighbor
        jidx = NL[i * M + j];

        // Look up x,y,z positions
        j_x = position_x[jidx];
        j_y = position_y[jidx];
        j_z = position_z[jidx];
        // Calc distance
        delx = i_x - j_x;
        dely = i_y - j_y;
        delz = i_z - j_z;
        r2inv = 1.0/( delx*delx + dely*dely + delz*delz );
        // Assume no cutoff and aways account for all nodes in area
        r6inv = r2inv * r2inv * r2inv;
        potential = r6inv*(1.5*r6inv - 2.0);
        // Sum changes in force
        force = r2inv*potential;
        fx += delx * force;
        fy += dely * force;
        fz += delz * force;
      }

      //Update forces after all neighbors accounted for.
      force_x[i] = fx;
      force_y[i] = fy;
      force_z[i] = fz;
    }
  }
}

double p_x[N], p_y[N], p_z[N];
double f_x[N], f_y[N], f_z[N];
int64_t NL[N * M];

int main() {
  for (int i = 0; i < N; ++i) {
    p_x[i] = rand();
    p_y[i] = rand();
    p_z[i] = rand();
    f_x[i] = rand();
    f_y[i] = rand();
    f_z[i] = rand();
  }
  for (int i = 0; i < N * M; ++i)
    NL[i] = rand() % N;

  md_kernel(p_x, p_y, p_z, f_x, f_y, f_z, NL);
  begin_roi();
  md_kernel(p_x, p_y, p_z, f_x, f_y, f_z, NL);
  end_roi();
  sb_stats();

  return 0;
}
