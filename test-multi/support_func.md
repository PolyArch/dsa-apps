## Multi-core ssim functions/primitives supported


* SS_REM_PORT(output_port, num_elem, mask, remote_port)

See fix_broadcast.c (looks like the network is giving a throughput of 64-bytes/cycle?)

* SS_IND_REM_SCRATCH(val_port, addr_port, num_elem, scr_base_addr, scratch_type)

* SS_REM_SCRATCH(scr_base_addr, stride, access_size, num_strides, val_port,
    scratch_type) - copy from a port to a global scratchpad location
See fix_remote_scr.c

* SS_WAIT_DF(num_rem_writes, scratch_type)

See fix_remote_scr.c

* SS_SCR_REM_SCR(src_scr_base_addr, stride, access_size, num_strides, dest_scr_base_addr, scratch_type)

See ...............

* SS_SCR_REM_PORT(scr_base_addr, num_strides, mask, remote_port)

See fix_scr_mem_scr.c

* SS_ATOMIC_SCR_OP(addr_port, val_port, offset, iters, opcode)

You can see in test-single

* SS_GLOBAL_BARRIER

TODO: add a test
There is a problem: currently it waits on the number of threads=constant
defined in the simulator.
Easier fix could be a parameter which would specify the number of threads to
wait (note it doesn't take into consideration which cores we should wait --
although this can be easily implemented)

# Utility functions in common/include/net_util_func.h

* addDest(mask, d): add multicast dest
* getLinearOffset(): global address for the location on a given core, offset in
                     the linear scratchpad

# Multi-core specific statistics

NETWORK: represents the bytes sent to the network
NETWORK_INC: bytes incoming from the network through SPU remote comm (TODO)
spu_link_utilization: TODO
throttle link utilization: for memory requests
avg_stall_time
avg_bug_msgs

# DEBUGGING DETAILS

* NET_REQ=1: to see cycle-level send and receive packets
* Remote: for each port in case of port mismatch error

# Run multiple times

* Way1: create threads multiple times

# changing bandwidth

link_bandwidth: ruby/network/BasicLink.py
mem_bw: SimpleMemory.py
SerialLink.py: ?
BasicLink.py: bandwidth_factor = 8 (to change spu link bw)
Spu message entries declared here: mem/protocol/MI_example-msg.sm 

# Results

For broadcast, its fully pipelined, rate is limited by the memory bandwidth
(just getting 1 byte/cycle?)

# HACKS (need information of number of cores (machinetype_num, and for threads, pass through SS_GLOBAL_WAIT(N)

* num_threads in ssim.cc
* required for global barrier in src/sim/process.hh

# Current limitations/Known bugs

* With begin_roi() and end_roi() switched on only once, it gives error for
running multiple times. And when used for each run, it gives twice the number of
cycles. Should it be changed?

* panic: Packet queue system.ruby.dir_cntrl0.memory- has grown beyond 100 packets -- need some backpressure
don't send prefetch request if number of awaiting memory requests > x

* Currently, it works only for MessageSizeType_Control. Seems like it consumes
  only 8-byte bandwidth. For this reason, fix_broadcast.c is able to give
  a throughput of 64 bytes/cycle. (TODO -- if can't fix, can split packets on
  simulator side as well!) 

* Size of linear and scratchpad size has to be same (Also address bits are max
    16 I guess?)


# Bottlenecks

* Memory bandwidth
* CGRA throughput
* Network bandwidth

## DGRA things (current status) -- TODO

It should be working but need to verify; add certain tests?
