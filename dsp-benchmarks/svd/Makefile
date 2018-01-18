ifndef N
N=32
endif


DFG = 

SCALE=$(N)

MACROS=-D N=$(N)

SOFTBRAINS = sb-origin

include ../common/Makefile.inc

sb-origin.o: sb-origin.cc $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3
