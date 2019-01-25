ifndef PRINT_RESULT
PRINT_RESULT=false
endif

ifndef C
C=1
endif

ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=main
TESTC=$(TESTS:=.cpp)
TEST_OBJS=$(TESTS:=.o)

all: $(TESTS)

# CPP=riscv64-unknown-elf-g++
# CC=riscv64-unknown-linux-gnu-gcc
CC=riscv64-unknown-linux-gnu-g++
CPP=riscv64-unknown-linux-gnu-g++

MACROS= -DC=$(C) -Dprint_result=$(PRINT_RESULT) -Ddense_act_file=$(dense_act_file) -Dwgt_ptr_file=$(wgt_ptr_file) -Dwgt_val_file=$(wgt_val_file) -Dwgt_ind_file=$(wgt_ind_file) -DN=$(N) -DM=$(M) -Dlayer_name=$(layer_name)
GCC=g++

DFGS=eie.dfg sparsify.dfg
# small_sp.dfg 

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

input_activations.data: gen_activations.py 
	python $< $(M) $(act_sp)

dense_activations.data: gen_dense_act.py 
	python $< $(M) $(act_sp)

input_weights.data: gen_weights.py 
	python $< $(N) $(M) $(syn_sp)

cpu: cpu.cpp input_activations.data input_weights.data dense_activations.data
	$(GCC) $< $(LIB) $(CFLAGS) $(MACROS) -o $(@) $(LFLAGS)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched --verbose $(SBCONFIG) $< 

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS) # input_activations.data input_weights.data dense_activations.data
	 $(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)
	
# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	 $(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --l2_size=4096kB --caches --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

run:
	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --l2_size=4096kB --caches --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

cleandfg:
	rm -f *.dfg.h
	rm -r sched

cleandata:
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) *.s *.o
	rm -r m5out stats verif viz gams

ultraclean: cleandata cleancpu clean
	rm -f *.log
