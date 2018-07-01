#!/bin/bash
#export SBCONFIG=$SS_TOOLS/configs/diannao_simd64.sbmodel
#export SBCONFIG=$SS_TOOLS/configs/revel.sbmodel
#export SBCONFIG=$SS_TOOLS/configs/spu_merge_test.sbmodel
gem5.opt $SS_STACK/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=softbrain


#SB_COMMAND = 1
#SB_ACC = 1
# row==j error (when .sbmodel is different for scheduler and gem5)
