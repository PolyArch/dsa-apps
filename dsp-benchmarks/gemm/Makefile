# It is somekind of cheating: The layout of matrix matters.
# I remove the overhead of rearranging data and suppose the
# data is already provided in this layout.
ifndef N
N=12
endif

ifndef M
M=16
endif

ifndef P
P=64
endif


DFG = compute.dfg

SOFTBRAINS = sb-origin sb-new sb-latency

SCALE=$(N) $(M) $(P)

MACROS=-D_N_=$(N) -D_M_=$(M) -D_P_=$(P)

ifneq ($(shell test $(N) -gt 10; echo $$?), 0)
MACROS += -D LATENCY
endif

include ../common/Makefile.inc

# The GeMM implemented in both old Softbrain and REVEL.
sb-latency.o sb-origin.o sb-new.o: %.o: %.cc compute.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
