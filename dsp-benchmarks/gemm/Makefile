ifndef N
N=120
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

MACROS=-D_N_=$(N) -D_M_=$(M) -D_P_=$(P)

include ../common/Makefile.inc

# The cholesky implemented in origin Softbrain. The comparison baseline.
sb-origin.o sb-new.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
