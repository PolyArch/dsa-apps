ifndef N
N=64
endif

DFG = compute.dfg subadd.dfg rotate.dfg

SOFTBRAINS = sb-origin sb-new sb-rec

SCALE=$(N)

MACROS=-D_N_=$(N)

include ../common/Makefile.inc

# sb-origin: The simplest origin Softbrain.
# sb-new: Softbrain with new instructions to remove control overhead.
sb-origin.o: %.o: %.cc compute.dfg.h subadd.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-new.o: %.o: %.cc rotate.dfg.h subadd.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
