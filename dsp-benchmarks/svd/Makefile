ifndef N
N=32
endif


DFG = vmc.dfg vv.dfg mvc.dfg vvc.dfg vm.dfg lmm2x2.dfg

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin

include ../common/Makefile.inc

sb-origin.o: sb-origin.cc vmc.h vv.h mvc.h vvc.h vm.h lmm2x2.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
