ifndef N
N=12
endif

SOFTBRAINS = sb-origin sb-new

DFG = dot.dfg mulconj.dfg finalize.dfg multi.dfg fused1.dfg fused2.dfg
SCALE=$(N)

MACROS=-D_N_=$(N)

include ../common/Makefile.inc

SOFTBRAINS = sb-origin

# original softbrain
sb-origin.o: sb-origin.cc mulconj.dfg.h finalize.dfg.h dot.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

sb-new.o: sb-new.cc multi.dfg.h fused1.dfg.h fused2.dfg.h
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
