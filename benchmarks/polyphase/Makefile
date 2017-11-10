ifndef N
N=10
endif

ifndef FILTER
FILTER=5
endif 

ifndef DECIMATION
DECIMATION=3
endif 

DFG=compute.dfg mul.dfg

SCALE=$(N) $(FILTER) $(DECIMATION)

MACROS=-D N=$(N) -D FILTER=$(FILTER) -D DECIMATION=$(DECIMATION)

include ../common/Makefile.inc

