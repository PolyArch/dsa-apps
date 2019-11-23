ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
# Change the default hardware here!
SBCONFIG=$(SS)"/ss-scheduler/configs/revel-1x1.sbmodel"
endif

# Change DFGs here!
DFG = dfg.dfg
DFG_HEADERS = $(DFG:.dfg=.dfg.h)


$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/ss_sched -a sa --max-iters 20000 -d $(FIFO_DEPTH) --verbose $(SBCONFIG) $< 

SRC = main.cc
EXE = $(SRC:.cc=.out)
LOG = $(SRC:.cc=.log)

CC = riscv64-unknown-elf-g++
CFLAGS = -O3 -std=c++14
# DO NOT use absolute paths to include files! USE -I flags!
IFLAGS = -I$(SS_TOOLS)/../ss-workloads/common/include -I$(SS_TOOLS)/include/

$(EXE): $(SRC) $(DFG_HEADERS)
	$(CC) $^ $(IFLAGS) $(CFLAGS) -o $(@)

$(LOG): $(EXE)
	SBCONFIG=$(SBCONFIG) BACKCGRA=1                                                       \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=MinorCPU                    \
	--l1d_size=2048kB --l1d_assoc=32 --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@

clean:
	rm -rf *.log *.out *.dfg.h m5out gams sched stats verif viz
