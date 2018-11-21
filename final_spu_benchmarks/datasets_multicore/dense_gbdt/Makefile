ifndef ratio
ratio=0.25
endif

ifndef C
C=4
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
CC=riscv64-unknown-linux-gnu-g++
CPP=riscv64-unknown-linux-gnu-g++


MACROS = -DN=$(N) -DM=$(M) -Dratio=$(ratio) -Dfile=$(file)

# DFGS=test.dfg final_redn.dfg map.dfg stall_none.dfg
DFGS=gbdt.dfg local_redn.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

input.data inst_id.data: gen.py
	python $< $(N) $(Mt) $(k) $(ratio)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched $(SBCONFIG) $<

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS) inst_id.data
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
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) *.s *.o
	rm -r m5out stats sched verif viz gams

ultraclean: cleandata cleancpu clean
	rm -f *.log
