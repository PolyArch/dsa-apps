ifndef N
N=12
endif

DFG = compute.dfg writeback.dfg multi.dfg

SCALE=$(N)

MACROS=-D_N_=$(N)

SOFTBRAINS = sb-origin sb-new sb-spec

include ../common/Makefile.inc

# The cholesky implemented in origin Softbrain. The comparison baseline.
sb-origin.o: sb-origin.cc compute.dfg.h writeback.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# On REVEL, using multi-DFG-single-CGRA tech gives pretty promissing
# acceleration.
sb-new.o: sb-new.cc multi.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

