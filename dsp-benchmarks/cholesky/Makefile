ifndef N
N=12
endif

DFG = compute.dfg writeback.dfg compute_dual.dfg offload.dfg finalize.dfg

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin sb-new sb-scr sb-spec sb-multi

include ../common/Makefile.inc

# The cholesky implemented in origin Softbrain. The comparison baseline.
sb-origin.o: sb-origin.cc compute.dfg.h writeback.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# The cholesky implemented in Softbrain with new control instructions.
# Also use recurrence to get rid of unnecessary induction memory IO.
sb-new.o: sb-new.cc compute_dual.dfg.h finalize.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# Based on the last one, move memory IO to spad IO.
sb-scr.o: sb-scr.cc compute.dfg.h writeback.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# Based on the last one, add data streaming between CGRAs.
sb-multi.o: sb-multi.cc compute_dual.dfg.h writeback.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# Based on the last one, add aggressive FU, like sqrter and 1/x divider to CGRA
sb-spec.o: sb-spec.cc compute_dual.dfg.h writeback.dfg.h offload.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

