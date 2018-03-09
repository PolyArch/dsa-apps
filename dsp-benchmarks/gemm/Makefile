# It is somekind of cheating: The layout of matrix matters.
# I remove the overhead of rearranging data and suppose the
# data is already provided in this layout.
ifndef N
N=48
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

# The GeMM implemented in both old Softbrain and REVEL.
sb-origin.o sb-new.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
