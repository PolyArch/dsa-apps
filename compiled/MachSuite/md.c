/*
Implemenataion based on:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010.
*/

#include "../Common/test.h"
#include "../Common/timing.h"
#include "../Common/interface.h"
#include "../Specs/md.h"

#ifndef U
#define U 1
#endif

struct Arguments {
  DTYPE p_x[N], p_y[N], p_z[N];
  DTYPE f_x[N], f_y[N], f_z[N];
  ITYPE NL[N * M];
  DTYPE p_x_[N], p_y_[N], p_z_[N];
  DTYPE f_x_[N], f_y_[N], f_z_[N];
  ITYPE NL_[N * M];
} args_;

void md_kernel(DTYPE force_x[],
               DTYPE force_y[],
               DTYPE force_z[],
               DTYPE position_x[],
               DTYPE position_y[],
               DTYPE position_z[],
               ITYPE NL[])
{

  #pragma ss config
  {
    DTYPE delx, dely, delz, r2inv;
    DTYPE r6inv, potential, force, j_x, j_y, j_z;
    DTYPE i_x, i_y, i_z, fx, fy, fz;
    ITYPE j, jidx;
    int64_t i;

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
        r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
        // Assume no cutoff and aways account for all nodes in area
        r6inv = r2inv * r2inv * r2inv;
        potential = r6inv * (1.5 * r6inv - 2.0);
        // Sum changes in force
        force = r2inv * potential;
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

struct Arguments *init_data() {
  DTYPE *p_x = args_.p_x;
  DTYPE *p_y = args_.p_y;
  DTYPE *p_z = args_.p_z;
  DTYPE *p_x_ = args_.p_x_;
  DTYPE *p_y_ = args_.p_y_;
  DTYPE *p_z_ = args_.p_z_;
  DTYPE *f_x = args_.f_x;
  DTYPE *f_y = args_.f_y;
  DTYPE *f_z = args_.f_z;
  DTYPE *f_x_ = args_.f_x_;
  DTYPE *f_y_ = args_.f_y_;
  DTYPE *f_z_ = args_.f_z_;
  ITYPE *NL = args_.NL;
  ITYPE *NL_ = args_.NL_;
  for (int i = 0; i < N; ++i) {
    p_x_[i] = p_x[i] = rand();
    p_y_[i] = p_y[i] = rand();
    p_z_[i] = p_z[i] = rand();
    f_x_[i] = f_x[i] = rand();
    f_y_[i] = f_y[i] = rand();
    f_z_[i] = f_z[i] = rand();
  }
  for (int i = 0; i < N * M; ++i)
    NL_[i] = NL[i] = rand() % N;
  return &args_;
}

void run_accelerator(struct Arguments *args, int iswarm) {
  DTYPE *p_x = args_.p_x;
  DTYPE *p_y = args_.p_y;
  DTYPE *p_z = args_.p_z;
  DTYPE *p_x_ = args_.p_x_;
  DTYPE *p_y_ = args_.p_y_;
  DTYPE *p_z_ = args_.p_z_;
  DTYPE *f_x = args_.f_x;
  DTYPE *f_y = args_.f_y;
  DTYPE *f_z = args_.f_z;
  DTYPE *f_x_ = args_.f_x_;
  DTYPE *f_y_ = args_.f_y_;
  DTYPE *f_z_ = args_.f_z_;
  ITYPE *NL = args_.NL;
  ITYPE *NL_ = args_.NL_;
  if (iswarm) {
    md_kernel(p_x_, p_y_, p_z_, f_x_, f_y_, f_z_, NL_);
  } else {
    md_kernel(p_x, p_y, p_z, f_x, f_y, f_z, NL);
  }
}

NO_SANITY_CHECK

