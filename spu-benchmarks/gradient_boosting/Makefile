ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

DFGS=none.dfg test.dfg final_redn.dfg map.dfg

DFG_HEADERS=$(DFGS:.dfg=.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cpp $(DFG_HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
