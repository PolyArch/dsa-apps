ifndef N
N=128
endif

DFG = compute0.dfg compute1.dfg

SCALE=$(N)

MACROS=-D N=$(N) -D VEC_WIDTH=$(VEC_WIDTH)

CASES=64 128 256 512 1024 2048

include ../common/Makefile.inc

