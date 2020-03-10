ifndef N
N=12
endif

ifndef LANES
LANES=8
endif

SOFTBRAINS = sb-origin sb-new sb-access sb-hetro sb-conc sb-latency sb-ds

DFG = \
  nmlz.dfg finalize.dfg norm.dfg hhr.dfg fused.dfg vcmm.dfg loader.dfg \
  rdfg.dfg qdfg.dfg l_rdfg.dfg l_qdfg.dfg temporal0.dfg temporal1.dfg

SCALE=$(N)

MACROS=-D_N_=$(N) -DLANES=$(LANES)

include ../common/Makefile.inc

# original softbrain
sb-origin.o sb-access.o: %.o: %.cc finalize.dfg.h nmlz.dfg.h norm.dfg.h vcmm.dfg.h loader.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-hetro.o sb-new.o sb-conc.o: %.o: %.cc rdfg.dfg.h qdfg.dfg.h loader.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-latency.o: %.o: %.cc l_rdfg.dfg.h l_qdfg.dfg.h qdfg.dfg.h loader.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-ds.o: sb-ds.cc temporal0.dfg.h temporal1.dfg.h loader.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
