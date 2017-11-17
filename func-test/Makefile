ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(RISCV)/configs/diannao_simd64.sbmodel
endif

TESTN=$(wildcard fix*.c)
TESTALL=$(wildcard *.c)
TESTC=$(filter-out $(TESTN), $(TESTALL))

TESTSN=$(patsubst %.c,bin/%,$(TESTN))
TESTS4=$(patsubst %.c,bin/%_4,$(TESTC))
TESTS12=$(patsubst %.c,bin/%_12,$(TESTC))
TESTS32=$(patsubst %.c,bin/%_32,$(TESTC))
TESTS500=$(patsubst %.c,bin/%_500,$(TESTC))
TESTS2048=$(patsubst %.c,bin/%_2048,$(TESTC))

TESTS=$(TESTS4) $(TESTS8)    $(TESTS12)   $(TESTS16) $(TESTS20) $(TESTS24) $(TESTS28) $(TESTS32) $(TESTS500) $(TESTS128)  $(TESTS2048) $(TESTSN) 

all: bin $(TESTS) 

CPP=riscv64-unknown-elf-g++

DFGS=$(wildcard *.dfg)
DFG_HEADERS=$(DFGS:.dfg=.h)
HEADERS=testing.h check.h
#${info $(DFG_HEADERS)}

bin:
	mkdir -p bin

$(DFG_HEADERS): %.h: %.dfg
	$(RISCV)/bin/sb_sched $(SBCONFIG) $<

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -static
#--std=c++11 

$(TESTSN):  bin/% : %.c $(DFG_HEADERS) $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=2048

$(TESTS4): bin/%_4 : %.c  $(DFG_HEADERS) $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=4 
$(TESTS12): bin/%_12 : %.c  $(DFG_HEADERS) $(HEADERS) 
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=12
$(TESTS32): bin/%_32 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=32
$(TESTS500): bin/%_500 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=500
$(TESTS2048): bin/%_2048 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -o $@ -DASIZE=2048


clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
	rm -rf m5out viz verif stats gams bin
