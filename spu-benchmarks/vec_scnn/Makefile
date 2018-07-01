ifndef Tx
Tx=28
endif

ifndef Ni
Ni=2
endif

ifndef Tn
Tn=2
endif

ifndef neuron_sp
neuron_sp=0.25
endif

ifndef synapse_sp
synapse_sp=0.5
endif

ifndef rle_width
rle_width=4
endif

ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++
MACROS= -DTx=$(Tx) -DNi=$(Ni) -DTn=$(Tn) -Dneuron_sp=$(neuron_sp) -Dsynapse_sp=$(synapse_sp) -Drle_width=$(rle_width)  
GCC=g++

DFGS=test.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

input_neuron.data: gen_neuron.py 
	python $< $(Tx) $(neuron_sp) $(rle_width) $(Ni)

input_synapse.data: gen_synapse.py 
	python $< $(Tx) $(neuron_sp) $(Ni) $(Tn)

cpu: cpu.cpp
	$(GCC) $< $(LIB) $(CFLAGS) $(MACROS) -o $(@) $(LFLAGS)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

$(TESTS): % :%.cpp $(DFG_HEADERS) input_neuron.data input_synapse.data cpu
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)
	
cleandata:
	rm -f input_neuron.data input_synapse.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o *.data
