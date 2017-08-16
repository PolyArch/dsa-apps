ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

TESTC=$(wildcard *.c)
TESTS4=$(patsubst %.c,bin/%_4,$(TESTC))
TESTS12=$(patsubst %.c,bin/%_12,$(TESTC))
TESTS32=$(patsubst %.c,bin/%_32,$(TESTC))
TESTS36=$(patsubst %.c,bin/%_36,$(TESTC))
TESTS2048=$(patsubst %.c,bin/%_2048,$(TESTC))

TESTS=$(TESTS4) $(TESTS8)    $(TESTS12)   $(TESTS16) $(TESTS20) $(TESTS24) $(TESTS28) $(TESTS32) $(TESTS36) $(TESTS128)  $(TESTS2048)

all: bin $(TESTS) 

CPP=riscv64-unknown-elf-g++

DFGS=$(wildcard *.dfg)
DFG_HEADERS=$(DFGS:.dfg=.h)
HEADERS=testing.h check.h
#${info $(DFGS)}
#${info $(DFG_HEADERS)}

bin:
	mkdir -p bin

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< sg

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 


$(TESTS4): bin/%_4 : %.c  $(DFG_HEADERS) $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@ -DASIZE=4 
$(TESTS12): bin/%_12 : %.c  $(DFG_HEADERS) $(HEADERS) 
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@ -DASIZE=12
$(TESTS32): bin/%_32 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@ -DASIZE=32
$(TESTS36): bin/%_36 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@ -DASIZE=36
$(TESTS2048): bin/%_2048 : %.c  $(DFG_HEADERS)  $(HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@ -DASIZE=2048


clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
