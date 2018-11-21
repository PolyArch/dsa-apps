ifndef ratio
ratio=0.4
endif

ifndef T
T=4
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
CC=riscv64-unknown-linux-gnu-gcc
CPP=riscv64-unknown-linux-gnu-g++

DFGS=ksvm.dfg duality_gap.dfg eta.dfg
MACROS= -DN=$(N) -DM=$(M) -Dratio=$(ratio) -DT=$(T) -Dfile=$(file)

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

input.data: gen.py
	python $< $(M) $(N) $(ratio)

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched --verbose $(SBCONFIG) $<

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS)
	 $(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)
	
# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS) $(DFG_HEADERS)
	 $(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(T) --num-dirs=$(T) --network=simple --topology=Mesh_XY --mesh-rows=$(T) --cmd=$(TESTS)

run:
	BACKCGRA=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=$(T) --num-dirs=$(T) --network=simple --topology=Mesh_XY --mesh-rows=$(T) --cmd=$(TESTS)

cleandfg:
	rm -f *.dfg.h

cleandata:
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) *.s *.o
	rm -r m5out stats sched verif viz gams

ultraclean: cleandata cleancpu clean
	rm -f *.log
