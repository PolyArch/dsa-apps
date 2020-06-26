ifndef N
N=12
endif

DFG = compute.dfg multi.dfg limited.dfg temporal.dfg

SCALE=$(N)

MACROS=-D_N_=$(N)

SOFTBRAINS = sb-origin sb-new sb-conc sb-access sb-limited sb-idiom sb-ds

include ../common/Makefile.inc

sb-origin.o sb-access.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-new.o sb-conc.o: %.o: %.cc multi.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-limited.o sb-idiom.o: %.o: %.cc limited.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-ds.o: sb-ds.cc temporal.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
