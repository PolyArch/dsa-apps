ifndef N
N=16
endif

ifndef FILTER
FILTER=8
endif 

DFG=compute.dfg mul.dfg

SCALE=$(N) $(FILTER)
MACROS=-D N=$(N) -D FILTER=$(FILTER)

include ../common/Makefile.inc

