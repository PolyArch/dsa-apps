ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

DFGS=none.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

TESTS=main
TESTC=$(TESTS:=.c)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -I$(SS)/ss-workloads/common/include
#--std=c++11 

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

$(TESTS): % :%.c $(DFG_HEADERS)
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@

run-classic:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=4096kB --l1i_size=16kB --l2_size=4096kB --caches --cmd=$(TESTS)

run-ruby:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=4 --num-dirs=4 --network=simple --topology=Mesh_XY --mesh-rows=2 --cmd=$(TESTS)

cleandfg:
	$(DFG_HEADERS)

clean:
	rm -f $(TESTS) *.s *.o

ultraclean:
	rm -f $(DFG_HEADERS) $(TESTS) *.s *.o


# gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=4 --num-dirs=4 --network=simple --topology=Mesh_XY --mesh-rows=2 --vecs-per-net=1 --routing-algorithm=1 --cmd=cpu
# $(SS_STACK)/gem5/build/RISCV/gem5.opt ~/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu
