ifndef N
N=4
endif


DFG = kernel1.dfg kernel2.dfg

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin

include ../common/Makefile.inc

sb-origin.o: sb-origin.cc kernel1.h kernel2.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
