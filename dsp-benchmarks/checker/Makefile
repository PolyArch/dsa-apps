ifndef N
N=32
endif


DFG = temporal.dfg

SCALE=$(N)

MACROS=-D _N_=$(N)

SOFTBRAINS = sb-origin sb-new sb-access

include ../common/Makefile.inc

sb-new.o: sb-new.cc temporal.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
