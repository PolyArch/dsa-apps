ifndef d1
d1=1
endif

ifndef d2
d2=8
endif

ifndef ratio
ratio=0.5
endif

ifndef C
C=8
endif

ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=main
TESTC=$(TESTS:=.cpp)
TEST_OBJS=$(TESTS:=.o)

all: $(TESTS)

#CPP=riscv32-unknown-elf-g++
# CPP=riscv64-unknown-elf-g++
# CC=riscv64-unknown-linux-gnu-gcc
CC=riscv64-unknown-linux-gnu-g++
CPP=riscv64-unknown-linux-gnu-g++

DFGS=ac.dfg
#test.dfg fwd_prop.dfg 
#vec_test.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

MACROS= -Dd1=$(d1) -Dd2=$(d2) -Dratio=$(ratio) -DC=$(C)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

input.data: complete_gen.py
	python $< $(d1) $(d2) $(ratio) 

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched --verbose $(SBCONFIG) $< 

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

# $(TEST_OBJS): $(TESTC) $(DFG_HEADERS) final_circuit.data final_index.data final_shadow_index.data
$(TEST_OBJS): $(TESTC) $(DFG_HEADERS)
	$(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)
	
# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	$(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

run:
	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

cleandfg:
	rm -f *.dfg.h

cleandata:
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
	rm -r m5out stats sched verif viz gams

ultraclean: cleandata cleancpu clean
	rm -f *.log
