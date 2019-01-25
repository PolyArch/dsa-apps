ifndef N
N=10
endif

ifndef M
M=100
endif

ifndef ratio
ratio=0.4
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

DFGS=new_err.dfg duality_gap.dfg eta.dfg
MACROS= -DN=$(N) -DM=$(M) -Dratio=$(ratio)

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

input.data: gen.py
	python $< $(M) $(N) $(ratio)


$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cpp $(DFG_HEADERS) input.data
	 $(CPP) $< $(MACROS) $(LIB) $(CFLAGS) -static -o $@

run:
	BACKCGRA=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --l2_size=4096kB --caches --cmd=$(TESTS)

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

cleandata:
	rm -f *.data

ultraclean: clean cleandata
