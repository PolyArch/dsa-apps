ifndef N
N=12
endif


DFG = vmc.dfg vv.dfg mvc.dfg vvc.dfg vm.dfg lmm2x2.dfg

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin sb-new

include ../common/Makefile.inc

sb-new.o sb-origin.o: %.o: %.cc vmc.h vv.h mvc.h vvc.h vm.h lmm2x2.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
