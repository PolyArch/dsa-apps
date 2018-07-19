ifndef d1
d1=1
endif

ifndef d2
d2=8
endif

ifndef ratio
ratio=0.5
endif


ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

#CPP=riscv32-unknown-elf-g++
CPP=riscv64-unknown-elf-g++

DFGS=test.dfg fwd_prop.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)
MACROS= -Dd1=$(d1) -Dd2=$(d2) -Dratio=$(ratio)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

input.data: complete_gen.py
	python $< $(d1) $(d2) $(ratio) 

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cpp $(DFG_HEADERS) input.data
	 $(CPP) $< $(MACROS) $(LIB) $(CFLAGS) -static -o $@

cleandata:
	rm -f *.data

cleandfg:
	rm -f $(DFG_HEADERS)

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean: clean cleandata
