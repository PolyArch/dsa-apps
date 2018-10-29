ifndef C
C = 4
endif

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

TESTS=main
TESTC=$(TESTS:=.cpp)
TEST_OBJS=$(TESTS:=.o)

all: $(TESTS)

#CPP=riscv64-unknown-elf-g++
CC=riscv64-unknown-linux-gnu-gcc
CPP=riscv64-unknown-linux-gnu-g++

MACROS= -DTx=$(Tx) -DNi=$(Ni) -DTn=$(Tn) -Dneuron_sp=$(neuron_sp) -Dsynapse_sp=$(synapse_sp) -Drle_width=$(rle_width)  -DC=$(C)
GCC=g++

DFGS=scnn.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

input_neuron.data: gen_neuron.py 
	python $< $(Tx) $(neuron_sp) $(rle_width) $(Ni)

input_synapse.data: gen_synapse.py 
	python $< $(Kx) $(synapse_sp) $(Ni) $(Tn)

cpu: cpu.cpp input_neuron.data input_synapse.data
	$(GCC) $< $(LIB) $(CFLAGS) $(MACROS) -o $(@) $(LFLAGS)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS) input_neuron.data input_synapse.data
	 $(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)
	
# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	 $(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

run:
	BACKCGRA=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

cleandfg:
	rm -f *.dfg.h

cleandata:
	rm -f input_neuron.data input_synapse.data input_neuron.data input_synapse.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean: cleandata cleancpu clean
	rm -f *.log
