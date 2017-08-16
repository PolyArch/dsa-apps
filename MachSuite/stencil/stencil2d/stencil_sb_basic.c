#include "stencil.h"
#include "softbrain.hpp"
#include "../../../common/include/sb_insts.h"
//#include "../../common/include/sim_timing.h"

#include "stencil_sb.h"

//SB version of 2d stencil
void stencil_sb(TYPE orig[row_size * col_size],
  TYPE sol[row_size * col_size],
  TYPE filter[f_size]) {
  int r,c,k1,k2;
  
  SB_CONFIG(stencil_sb_config,stencil_sb_size);
  
  // yfl's code
  for (r = 0; r < row_size - 2; r++) {
    for (c = 0; c < col_size - 2; c++) {
      SB_CONST(P_stencil_sb_Carry, 0, 1);

      SB_DMA_READ(orig + r * col_size + c, 
        col_size * sizeof(TYPE),
        4 * sizeof(TYPE),
        3,
        P_stencil_sb_I);

      SB_DMA_READ(filter,
        3 * sizeof(TYPE),
        4 * sizeof(TYPE),
        3,
        P_stencil_sb_F);

      SB_RECURRENCE(P_stencil_sb_R, P_stencil_sb_Carry, 2);
       
      SB_DMA_WRITE(P_stencil_sb_R,
        sizeof(TYPE),
        sizeof(TYPE),
        1,
        sol + r * col_size + c);
    }
  }

  SB_WAIT_ALL();
}
