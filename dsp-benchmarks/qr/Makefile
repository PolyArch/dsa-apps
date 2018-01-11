ifndef N
N=8
endif

SOFTBRAINS = sb-origin sb-new sb-scr sb-multi

DFG = mulconj.dfg finalize.dfg conjconj.dfg finalize_compact.dfg
SCALE=$(N)

MACROS=-D N=$(N)

include ../common/Makefile.inc

SOFTBRAINS = sb-origin

# original softbrain
sb-origin.o: sb-origin.cc mulconj.h finalize.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

# New: Apply new softbrain technique data discard to eliminate control overhead
# SPad: Store the data on the spad
# Multi: Use SB_CONTEXT instead of SB_CONFIG
# FIXME: Spad verson only works on R. I do not know why Q is not correct.
sb-multi.o sb-new.o sb-scr.o: %.o: %.cc mulconj.h finalize_compact.h conjconj.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

