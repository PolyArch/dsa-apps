ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

TESTC=$(wildcard *.c)
TESTS=$(patsubst %.c,%,$(TESTC))

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

DFGS=$(wildcard *.dfg)
DFG_HEADERS=$(DFGS:.dfg=.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

$(DFG_HEADERS): %.h: %.dfg
	SBCONFIG=$(SBCONFIG) $(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $<

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3  -I../common/include/ -I$(SS_TOOLS)/include/
#--std=c++11 

$(TESTS): % :%.c $(DFG_HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
