/*
Implementations based on:
Harish and Narayanan. "Accelerating large graph algorithms on the GPU using CUDA." HiPC, 2007.
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

#include "bfs.h"
#include "bfs_sb.h"
#include "../../../common/include/sb_insts.h"


void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
            node_index_t starting_node, level_t level[N_NODES],
            edge_index_t level_counts[N_LEVELS])
{
  node_index_t n;
  edge_index_t e;
  level_t horizon;
  edge_index_t cnt;
  int total_size;

  level[starting_node] = 0;
  level_counts[0] = 1;

  SB_CONFIG(bfs_sb_config,bfs_sb_size);
  SB_STRIDE(8,8);

  //first iteration;
  n = starting_node;
  edge_index_t tmp_begin = nodes[n].edge_begin;
  edge_index_t tmp_end = nodes[n].edge_end;
  uint64_t size = tmp_end-tmp_begin;

  SB_DMA_READ_SIMP(&edges[tmp_begin].dst,size,P_IND_DOUB0);
  SB_INDIRECT64(P_IND_DOUB0,&level[0],size,P_bfs_sb_L);

  SB_CONST(P_bfs_sb_reset,0,size-1);
  SB_CONST(P_bfs_sb_reset,1,1); //RESET

  SB_CONST(P_bfs_sb_H,horizon,size);

  SB_GARBAGE_SIMP(P_bfs_sb_CNT,size-1);
  SB_DMA_WRITE(P_bfs_sb_CNT,8,8,1,&cnt); //Get value of output

  SB_INDIRECT64_WR(P_IND_DOUB1,&level[0],size,P_bfs_sb_NewL);

  SB_WAIT_ALL();

  if( (level_counts[horizon+1]=cnt)==0 )
    return;


  loop_horizons: for( horizon=1; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;
    //printf("horizon %d\n",horizon);
    // Add unmarked neighbors of the current horizon to the next horizon
    loop_nodes: for( n=0; n<N_NODES; ++n ) {
      if( level[n]==horizon ) {
        //SB_WAIT_ALL();

        //printf("Doing %d of %d nodes\n", n, N_NODES);
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        uint64_t size = tmp_end-tmp_begin;

        //printf("node %ld is on horizon, ",n);
        //printf("with %lld edges \n",size);

        SB_DMA_READ_SIMP(&edges[tmp_begin].dst,size,P_IND_DOUB0);
        SB_INDIRECT64(P_IND_DOUB0,&level[0],size,P_bfs_sb_L);

        SB_CONST(P_bfs_sb_reset,0,size);
        SB_CONST(P_bfs_sb_H,horizon,size);
        SB_GARBAGE_SIMP(P_bfs_sb_CNT,size);

        SB_INDIRECT64_WR(P_IND_DOUB1,&level[0],size,P_bfs_sb_NewL);

        //if((total_size&1)==0) { //EVEN NUMBER
        //  SB_WAIT_ALL();
        //}


        //printf("cnt_tmp = %lld, cnt = %lld\n",cnt_tmp, cnt);

        //Loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
        //  printf("%d ", level[edges[e].dst]);
        //}
        //Printf("\n");

        //loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ )        {
        //  node_index_t tmp_dst = edges[e].dst;
        //  level_t tmp_level = level[tmp_dst];

        //  if( tmp_level ==MAX_LEVEL ) { // Unmarked
        //    level[tmp_dst] = horizon+1;
        //    ++cnt;
        //  } else {
        //    level[tmp_dst] = tmp_level;
        //  }
        //}
      }
    }

    SB_WAIT_ALL();
    SB_CONST(P_bfs_sb_reset,1,1); //RESET
    SB_CONST(P_bfs_sb_H,1,1); //DUMMY H
    SB_CONST(P_bfs_sb_L,1,1); //DUMMY L
    SB_DMA_WRITE(P_bfs_sb_CNT,8,8,1,&cnt); //Get value of output
    SB_GARBAGE(P_bfs_sb_NewL,1); //toss the NewL
    SB_WAIT_ALL();

//    for(int i = 0; i < N_NODES; ++i) {
//      printf("%d ",level[i]);
//    }
//    printf("\n");
//
//
//    printf(" cnt = %lld\n",  cnt);
    if( (level_counts[horizon+1]=cnt)==0 )
      break;
  }
}
