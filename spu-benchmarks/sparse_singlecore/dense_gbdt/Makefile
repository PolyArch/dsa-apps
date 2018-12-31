ifndef ratio
ratio=0.25
endif

ifndef C
C=1
endif

ifndef N
N=10
endif

ifndef M
M=136
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

MACROS = -DN=$(N) -DM=$(M) -Dratio=$(ratio) -Dfile=$(file) -DC=$(C)

# DFGS=test.dfg final_redn.dfg map.dfg stall_none.dfg
DFGS=gbdt.dfg local_redn.dfg map.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

OPT?=-O0 #-O3
# CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
CFLAGS=$(OPT) -I$(SS)/ss-workloads/common/include -DOPENMP -fopenmp
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
	MAPPING=COL BACKCGRA=1 LINEAR_SCR=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --cmd=$(TESTS)

run:
	MAPPING=COL BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --cmd=$(TESTS)

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
