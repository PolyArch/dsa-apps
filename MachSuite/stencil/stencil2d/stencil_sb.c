#include "stencil.h"
#include "softbrain.hpp"
#include "../../../common/include/sb_insts.h"
#include "stencil_sb.h"

#define pipewidth 128

//SB version of 2d stencil
void stencil_sb(TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
  int r,c,k1,k2;
  
  SB_CONFIG(stencil_sb_config,stencil_sb_size);
  for (r=0; r<row_size-2; r++) {
    for (c=0; c+pipewidth<col_size-2+1; c+=pipewidth) {
      SB_CONST(P_stencil_sb_C,0,pipewidth);
      SB_RECURRENCE(P_stencil_sb_O, P_stencil_sb_C, 8*pipewidth);

      for (k1=0;k1<3;k1++){  //Row access
        for (k2=0;k2<3;k2++){  //column access
          //mul = filter[k1*3 + k2] * orig[(r+k1)*col_size + c+k2]; -- vectorize this pipewidth times
          SB_DMA_READ(&orig[(r+k1)*col_size+c+k2],  0, pipewidth*sizeof(TYPE), 1, P_stencil_sb_I);  
          SB_CONST(P_stencil_sb_F,filter[k1*3+k2], pipewidth/2);
        }
      }

      SB_DMA_WRITE(P_stencil_sb_O, 0, pipewidth*sizeof(TYPE), 1, &sol[(r*col_size)+c]); 
    } 
  }

  SB_WAIT_ALL();
}
