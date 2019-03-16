ifndef csr_file
# csr_file=\"example.csv\"
# csr_file=\"cycle_example.csv\"
# csr_file=\"datasets/directed_uniform/rome99_csr\"
csr_file=\"datasets/bcspwr01.mtx\"
endif

ifndef C
C=1
endif

ifndef V
V=39
endif

ifndef E
E=85
endif

include /home/vidushi/ss-stack/ss-workloads/spu-benchmarks/common/Makefile.inc

MACROS = -DV=$(V) -DE=$(E) -Dcsr_file=$(csr_file) -DC=$(C)

DFGS=pr.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched $(SBCONFIG) $<

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS) # inst_id.data
	 $(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)

# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	 $(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 LINEAR_SCR=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

# run-classic:
# 	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --num-cpus=$(C) --num-dirs=$(C) --cmd=$(TESTS)

run:
	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --ruby --num-cpus=$(C) --num-dirs=1 --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)
