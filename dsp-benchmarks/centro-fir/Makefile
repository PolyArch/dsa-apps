ifndef N
N=1024
endif

ifndef M
M=37
endif

DFG = compute.dfg

DFG_HEADERS=$(DFG:.dfg=.h)

SCALE = $(N) $(M)

SOFTBRAINS = sb-new sb-origin sb-spad

MACROS=-D_N_=$(N) -D_M_=$(M)

include ../common/Makefile.inc


# New instructions remove control overhead
sb-origin.o sb-new.o sb-spad.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
