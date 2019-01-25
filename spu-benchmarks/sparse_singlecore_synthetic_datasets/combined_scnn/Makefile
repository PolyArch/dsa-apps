ifndef Tx
Tx=32
#Tx=14
endif

ifndef Kx
Kx=3
endif


ifndef Ni
Ni=64
endif

ifndef Tn
Tn=1
endif

ifndef neuron_sp
neuron_sp=0.25
endif

ifndef synapse_sp
synapse_sp=0.45
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
	python $< $(Kx) $(synapse_sp) $(Ni) $(Tn)

cpu: cpu.cpp
	$(GCC) $< $(LIB) $(CFLAGS) $(MACROS) -o $(@) $(LFLAGS)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

$(TESTS): % :%.cpp $(DFG_HEADERS) input_neuron.data input_synapse.data
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

run:
	BACKCGRA=1 MAPPING=ROW gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --l2_size=4096kB --caches --cmd=$(TESTS)
	
cleandata:
	rm -f input_neuron.data input_synapse.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean: cleandata cleancpu clean
	rm -f *.log
