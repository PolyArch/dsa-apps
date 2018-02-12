ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/revel.sbmodel
endif

DFG_HEADERS = $(DFG:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I../../../ss-tools/include -I../common/
INTELI=-I /opt/intel/include 
CFLAGS=-O3 -std=c++11
LMKL=-L"/opt/intel/mkl/lib" -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm

CC=riscv64-unknown-elf-g++

BENCHMARKS = optimized
OBJS = $(BENCHMARKS:=.o)
LOGS = $(SOFTBRAINS:=.log)
EXEC = $(BENCHMARKS:=.exe) $(SOFTBRAINS:=.exe)

input.data ref.data: gen.py
	python $< $(SCALE)

$(OBJS): %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) $(CFLAGS)

physical.exe: main.cc optimized.cc
	g++ $^ -o $@ $(MACROS) $(IFLAGS) $(CFLAGS)

physical.log mkl.log: %.log: %.exe input.data ref.data
	./$^ 2>&1 | tee $@

mkl.exe: mkl.cc
	icc $^ -o $@ $(LMKL) $(CFLAGS) $(MACROS) $(IFLAGS) $(INTELI)
	

ooo.log: optimized.exe input.data ref.data
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=detailed --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@

$(LOGS): %.log: %.exe input.data ref.data
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@
	
$(EXEC): %.exe: %.o main.cc
	$(CC) main.cc $(basename $@).o -o $@ -lm $(CFLAGS) $(MACROS) $(IFLAGS)

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf *.dfg.h

clean: cleandata
	rm -f *.o *.log *.exe
	rm -rf stats gams m5out viz verif

ultraclean: clean cleandfg
	rm  -f *.csv
