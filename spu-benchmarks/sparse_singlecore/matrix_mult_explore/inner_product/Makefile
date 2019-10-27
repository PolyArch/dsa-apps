ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

ifndef N
N=64
endif

ifndef M
M=4096
endif

TESTS=main
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

# CPP=riscv64-unknown-elf-g++
CPP=riscv64-unknown-linux-gnu-g++
MACROS= -DN=$(N) -DM=$(M) -Dact_sp=$(act_sp) -Dsyn_sp=$(syn_sp) 
GCC=g++

DFGS=eie.dfg
#test.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

input_activations.data: gen_activations.py 
	python $< $(M) $(act_sp)

input_weights.data: gen_weights.py 
	python $< $(N) $(M) $(syn_sp)

cpu: cpu.cpp input_activations.data input_weights.data
	$(GCC) $< $(LIB) $(CFLAGS) $(MACROS) -o $(@) $(LFLAGS)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched --verbose $(SBCONFIG) $< 

$(TESTS): % :%.cpp $(DFG_HEADERS) input_activations.data input_weights.data
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

run:
	BACKCGRA=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --cmd=$(TESTS)
	
cleandata:
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean: cleandata cleancpu clean
	rm -f *.log
