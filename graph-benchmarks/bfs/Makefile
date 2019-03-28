ifndef data_file
# data_file=\"datasets/rome99_csr\"
data_file=\"datasets/fb_csr\"
endif

ifndef csr_file
# csr_file=\"datasets/bcspwr01.mtx\"
# csr_file=\"datasets/rome_99\"
csr_file=\"datasets/fb_99\"
endif

ifndef C
C=8
endif

ifndef NUM_THREADS
NUM_THREADS=2
endif

ifndef V
# V=39
# V=3353
# V=3352
V=50516
endif

ifndef E
# E=85
# E=8859
E=1638612
endif

include /home/vidushi/ss-stack/ss-workloads/spu-benchmarks/common/Makefile.inc

MACROS = -DV=$(V) -DE=$(E) -Dcsr_file=$(csr_file) -DC=$(C) -Ddata_file=$(data_file) -DNUM_THREADS=$(NUM_THREADS)

DFGS=bfs.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched $(SBCONFIG) $<

csr: make_unweighted.cpp
	g++ -std=c++11 make_unweighted.cpp $(MACROS) -o csr
	./csr

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
