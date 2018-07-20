/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"
#include "ellpack.dfg.h"
#include "../../../common/include/sb_insts.h"

void ellpack(TYPE nzval[N*L], int64_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i;

    SB_CONFIG(ellpack_config,ellpack_size);

    SB_DMA_READ(&cols[0], 0, L * sizeof(TYPE),N,P_IND_1);
    SB_INDIRECT(P_IND_1,&vec[0],N*L,P_ellpack_Vec);
    SB_DMA_READ(&nzval[0], 0, L * sizeof(TYPE),N,P_ellpack_Val);

    int steps_m1 = L/4-1; //2 lanes
    
    SB_STRIDE(8,8); //for making later commands simpler

    ellpack_1 : for (i=0; i<N; i++) {        
        SB_CONST(P_ellpack_reset,0,steps_m1);
        SB_CONST(P_ellpack_reset,1,1);
        SB_GARBAGE_SIMP(P_ellpack_O,steps_m1);
        SB_DMA_WRITE_SIMP(P_ellpack_O,1,&out[i]);

        //ellpack_2 : for (j=0; j<L; j++) {
        //        Si = nzval[j + i*L] * vec[cols[j + i*L]];
        //        sum += Si;
        //}
        //out[i] = sum;
    }

    SB_WAIT_ALL();

}
