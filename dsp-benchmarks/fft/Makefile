ifndef N
N=64
endif

DFG = compute0.dfg compute1.dfg compute_rec.dfg

SOFTBRAINS = sb-origin sb-new sb-scr sb-rec

SCALE=$(N)

MACROS=-D N=$(N)

include ../common/Makefile.inc

# sb-origin: The simplest origin Softbrain.
# sb-new: Softbrain with new instructions to remove control overhead.
# sb-scr: All the data will go through the spad instead of memory. TOO AGGRESSIVE, crashes on large scale.
sb-new.o sb-origin.o: %.o: %.cc compute0.dfg.h compute1.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

