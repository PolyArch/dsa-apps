/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"
#include "ellpack.dfg.h"
#include <ss-intrin/ss_insts.h>

void ellpack(TYPE nzval[N*L], int64_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i;

    SS_CONFIG(ellpack_config,ellpack_size);

    SS_DMA_READ(&cols[0], 0, L * sizeof(TYPE),N,P_IND_1);
    SS_CONFIG_INDIRECT(T64, T64, 8);
    SS_INDIRECT(P_IND_1,&vec[0],N*L,P_ellpack_Vec);
    SS_DMA_READ(&nzval[0], 0, L * sizeof(TYPE),N,P_ellpack_Val);

    int steps_m1 = L/4-1; //2 lanes
    
    ellpack_1 : for (i=0; i<N; i++) {        
        SS_CONST(P_ellpack_reset,0,steps_m1);
        SS_CONST(P_ellpack_reset,1,1);
        SS_GARBAGE(P_ellpack_O,steps_m1);
        SS_DMA_WR_INNER(&out[i], 8, P_ellpack_O);

        //ellpack_2 : for (j=0; j<L; j++) {
        //        Si = nzval[j + i*L] * vec[cols[j + i*L]];
        //        sum += Si;
        //}
        //out[i] = sum;
    }

    SS_WAIT_ALL();

}
