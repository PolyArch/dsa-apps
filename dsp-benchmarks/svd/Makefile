ifndef N
N=4
endif


DFG = hes_mul.dfg mm_sb.dfg mul_acc.dfg conj_mul_acc.dfg fin_a_mul_outer.dfg fin_outer_mul_a.dfg

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin

include ../common/Makefile.inc

sb-origin.o: sb-origin.cc $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O2
