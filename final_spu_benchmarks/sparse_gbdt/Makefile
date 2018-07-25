ifndef N
N=16
endif

ifndef Mt
Mt=4
endif

ifndef k
k=64
endif

ifndef ratio
ratio=0.25
endif

ifndef M
M=4
endif

ifndef dep_ratio
dep_ratio=0.25
endif


ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++
MACROS = -DN=$(N) -DMt=$(Mt) -Dk=$(k) -Dratio=$(ratio) -DM=$(M)

DFGS=sparse_none.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

input.data inst_id.data: gen.py
	python $< $(N) $(Mt) $(ratio) $(dep_ratio)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

$(TESTS): % :%.cpp $(DFG_HEADERS) input.data
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

run:
	BACKCGRA=1 MAPPING=COL gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=4096kB --l1i_size=16kB --l2_size=4096kB --caches --cmd=softbrain

run-debug:
	BACKCGRA=1 MAPPING=COL gdb --args gem5.debug $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=softbrain


cleandata:
	rm -f input.data inst_id.data

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o *.data
