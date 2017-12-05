#!/bin/bash
gem5.opt $SS_STACK/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=main


#gdb --args
#SB_COMMAND = 1
#SB_ACC = 1
