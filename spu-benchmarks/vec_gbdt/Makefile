ifndef N
N=100
endif

ifndef Mt
Mt=32
endif

ifndef k
k=64
endif

ifndef ratio
ratio=0.5
endif

ifndef M
M=8
endif


ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++
MACROS = -DN=$(N) -DMt=$(Mt) -Dk=$(k) -Dratio=$(ratio) -DM=$(M)

DFGS=none.dfg test.dfg final_redn.dfg map.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

input.data inst_id.data: gen.py
	python $< $(N) $(Mt) $(k) $(ratio)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

$(TESTS): % :%.cpp $(DFG_HEADERS) input.data
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

cleandata:
	rm -f input.data inst_id.data

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o *.data
