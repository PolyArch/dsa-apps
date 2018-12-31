ifndef ratio
ratio=0.25
endif

ifndef C
C=16
endif

ifndef N
N=10
endif

ifndef M
M=136
endif

include ../../common/Makefile.inc

MACROS = -DN=$(N) -DM=$(M) -Dratio=$(ratio) -Dfile=$(file) -DC=$(C)

# DFGS=test.dfg final_redn.dfg map.dfg stall_none.dfg
DFGS=gbdt.dfg local_redn.dfg map.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

input.data inst_id.data: gen.py
	python $< $(N) $(Mt) $(k) $(ratio)

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
	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)
