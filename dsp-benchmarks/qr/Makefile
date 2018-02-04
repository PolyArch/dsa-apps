ifndef N
N=12
endif

SOFTBRAINS = sb-origin sb-new sb-scr sb-wrap

DFG = dot.dfg norm.dfg mulconj.dfg finalize.dfg finalize_compact.dfg multi.dfg
SCALE=$(N)

MACROS=-D N=$(N)

include ../common/Makefile.inc

SOFTBRAINS = sb-origin

# original softbrain
sb-origin.o: sb-origin.cc mulconj.dfg.h finalize.dfg.h dot.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# New: Apply new softbrain technique data discard to eliminate control overhead
# SPad: Store the data on the spad
# Multi: Use SB_CONTEXT instead of SB_CONFIG
# FIXME: Spad verson only works on R. I do not know why Q is not correct.
sb-multi.o sb-new.o sb-scr.o: %.o: %.cc mulconj.dfg.h finalize_compact.dfg.h norm.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3


sb-wrap.o: sb-wrap.cc multi.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
