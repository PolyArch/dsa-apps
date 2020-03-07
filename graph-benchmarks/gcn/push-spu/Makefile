ifndef data_file
data_file=\"../../pagerank/datasets/rome99_csr\"
# data_file=\"example.csv\"
# data_file=\"gem5_rome99.txt\"
# data_file=\"lj_short.txt\"
# data_file=\"lj_larger.txt\"
# data_file=\"fb_gem5.txt\"
# data_file=\"flickr_gem5.txt\"
# data_file=\"roadny_gem5.txt\"
# data_file=\"col_gem5.txt\"
# data_file=\"lj_gem5_16.txt\"
endif

ifndef csr_file
csr_file=\"../../pagerank/datasets/bcspwr01.mtx\"
# csr_file=\"../../pagerank/datasets/rome99_csr\"
# csr_file=\"../../pagerank/datasets/rome_99\"
#
# csr_file=\"example.csv\"
# csr_file=\"gem5_rome99.txt\"
# csr_file=\"lj_short.txt\"
# csr_file=\"lj_larger.txt\"
# csr_file=\"fb_gem5.txt\"
# csr_file=\"flickr_gem5.txt\"
# csr_file=\"roadny_gem5.txt\"
# csr_file=\"col_gem5.txt\"
# data_file=\"lj_gem5_16.txt\"
endif

ifndef C
C=16
endif

ifndef V
V=820878
# V=39
# V=3353
# V=633278
# V=3352
# V=6328
# V=12664
# V=12624 # fb_gem5
# V=205216 # flickr_gem5
# V=66080 # roadny_gem5
# V=108912 # col_gem5
# V=6
endif

ifndef E
E=85
# E=8859
# E=7880940 # flickr_gem5
# E=9049
# E=8852
# E=8840
# E=16236
# E=34916
# E=103240 # fb_gem5
# E=178476 # roadny_gem5
# E=280628 # col_gem5
# E=12
# E=2843359
endif

ifndef SYNC
SYNC=0
endif

ifndef AGG
AGG=1
endif

ifndef MULT
MULT=1
endif

ifndef PRELOAD
PRELOAD=0
endif

ifndef POSTLOAD
POSTLOAD=0
endif

ifndef FULL_TRAINING
FULL_TRAINING=0
endif

ifndef LADIES_SAMPLE
LADIES_SAMPLE=256
endif

ifndef GCN_LAYERS
GCN_LAYERS=1
endif

ifndef NUM_THREADS
NUM_THREADS=8
endif

ifndef LOAD_BALANCE
LOAD_BALANCE=0
endif


include /home/vidushi/ss-stack/ss-workloads/spu-benchmarks/common/Makefile.inc

MACROS = -DV=$(V) -DE=$(E) -Dcsr_file=$(csr_file) -DC=$(C) -Ddata_file=$(data_file) -DSYNC=$(SYNC) -DAGG=$(AGG) -DMULT=$(MULT) -DPRELOAD=$(PRELOAD) -DPOSTLOAD=$(POSTLOAD) -DFULL_TRAINING=$(FULL_TRAINING) -DLADIES_SAMPLE=$(LADIES_SAMPLE) -DGCN_LAYERS=$(GCN_LAYERS) -DNUM_THREADS=$(NUM_THREADS) -DLOAD_BALANCE=$(LOAD_BALANCE)

DFGS=agg.dfg mult.dfg gcn.dfg

DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	ss_sched -v -a sa $(SBCONFIG) $<

csr: make_unweighted.cpp
	g++ -std=c++11 make_unweighted.cpp $(MACROS) -o csr
	./csr

sampling: cpu.cpp
	# rm sampled_nodes.txt
	# rm adj_mat.txt
	g++ -std=c++11 cpu.cpp $(MACROS) -o csr
	./csr

$(TEST_OBJS): $(TESTC) $(DFG_HEADERS) # inst_id.data
	 $(CC) -c $< $(LIB) $(CFLAGS) $(MACROS) -I. -static -o $(@)

# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	 $(CPP) $(LIB) $(CFLAGS) $(MACROS) -static -o $(@) $< $(LFLAGS)

run-debug:
	BACKCGRA=1 NET_REQ=1 COMMAND_I=1 COMMAND_O=1 COMP=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=$(C) --cmd=$(TESTS)

# run-classic:
# 	BACKCGRA=1 LINEAR_SCR=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=2048kB --l1i_size=16kB --caches --num-cpus=$(C) --num-dirs=$(C) --cmd=$(TESTS)

run:
	BACKCGRA=1 DSCDIVLD=1 gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1i_size=16kB --ruby --num-cpus=$(C) --num-dirs=$(C) --network=simple --topology=Mesh_XY --mesh-rows=2 --cmd=$(TESTS) --mem-size=4GB
