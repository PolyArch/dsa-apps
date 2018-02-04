ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

TESTC=$(wildcard *.c)
TESTS=$(patsubst %.c,%,$(TESTC))

all: $(TESTS) bin

CPP=riscv64-unknown-elf-g++

DFGS=$(wildcard *.dfg)
DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

$(DFG_HEADERS): %.dfg.h: %.dfg
	SBCONFIG=$(SBCONFIG) $(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $<

bin:
	mkdir -p bin

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3  -I../common/include/ -I$(SS_TOOLS)/include/
#--std=c++11 

$(TESTS): % :%.c $(DFG_HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o bin/$@

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
