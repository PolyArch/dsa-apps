ifndef N
N=1024
endif

DFG = sbdfg.dfg another.dfg

SOFTBRAINS = sb-origin sb-none sb-new sb-ultra

SCALE=$(N)

MACROS=-DN=$(N)

include ../common/Makefile.inc

# The cholesky implemented in origin Softbrain. The comparison baseline.
sb-none.o: %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-new.o sb-origin.o: %.o: %.cc sbdfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-ultra.o: %.o: %.cc another.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
