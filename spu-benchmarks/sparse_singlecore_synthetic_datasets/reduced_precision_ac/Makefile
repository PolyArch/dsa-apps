ifndef d1
d1=1
endif

ifndef d2
d2=8
endif

ifndef ratio
ratio=0.5
endif


ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=softbrain
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

#CPP=riscv32-unknown-elf-g++
CPP=riscv64-unknown-elf-g++

DFGS=fwd_prop.dfg
#test.dfg fwd_prop.dfg 
#vec_test.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)
MACROS= -Dd1=$(d1) -Dd2=$(d2) -Dratio=$(ratio)

#${info $(DFGS)}
#${info $(DFG_HEADERS)}

input.data: complete_gen.py
	python $< $(d1) $(d2) $(ratio) 

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/ss_sched --verbose $(SBCONFIG) $< 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cpp $(DFG_HEADERS) input.data
	 $(CPP) $< $(MACROS) $(LIB) $(CFLAGS) -static -o $@

run:
	BACKCGRA=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=4096kB --l1i_size=16kB --l2_size=4096kB --caches --cmd=softbrain

run-debug:
	BACKCGRA=1 gdb --args gem5.debug $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=softbrain



cleandata:
	rm -f *.data

cleandfg:
	rm -f $(DFG_HEADERS)

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o

ultraclean: clean cleandata
