ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

TESTS=main
TESTC=$(TESTS:=.c)

all: $(TESTS)

#CPP=riscv32-unknown-elf-g++
CPP=riscv64-unknown-elf-g++

DFGS=dot.dfg

DFG_HEADERS=$(DFGS:.dfg=.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

qr.o: qr_sb.c
	$(CPP) $<  $(CFLAGS) -c -o $@

$(TESTS): % :%.c $(DFG_HEADERS) qr.o
	 $(CPP) $< qr.o  $(LIB) $(CFLAGS) -static -o $@

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
	rm -rf viz stats verif gams
