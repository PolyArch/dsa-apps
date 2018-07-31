ifndef N
N=12
endif

DFG = compute.dfg multi.dfg

SCALE=$(N)

MACROS=-D_N_=$(N)

SOFTBRAINS = sb-origin sb-new

include ../common/Makefile.inc

sb-origin.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-new.o: %.o: %.cc multi.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

