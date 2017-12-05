ifndef N
N=12
endif

ifndef M
M=16
endif

ifndef P
P=64
endif

DFG = compute.dfg

SOFTBRAINS = sb-origin sb-new

SCALE=$(N) $(M) $(P)

MACROS=-DN=$(N) -DM=$(M) -DP=$(P)

include ../common/Makefile.inc

# The cholesky implemented in origin Softbrain. The comparison baseline.
sb-origin.o sb-new.o: %.o: %.cc compute.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O2
