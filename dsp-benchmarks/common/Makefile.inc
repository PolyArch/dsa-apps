ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/revel.sbmodel
endif

DFG_HEADERS = $(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I ../../../ss-tools/include -I ../common/ -I ./dfgs

CC=riscv64-unknown-elf-g++

BENCHMARKS = baseline optimized
OBJS = $(BENCHMARKS:=.o)
LOGS = $(BENCHMARKS:=.log) $(SOFTBRAINS:=.log)
EXEC = $(BENCHMARKS:=.exe) $(SOFTBRAINS:=.exe)

input.data ref.data: gen.py
	python $< $(SCALE)

$(OBJS): %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O3

physical.exe: main.cc optimized.cc
	g++ $^ -o $@ $(MACROS) $(IFLAGS) -std=c++11 -O3

physical.log: physical.exe input.data ref.data
	./physical.exe 2>&1 | tee $@
	

$(LOGS): %.log: %.exe
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@
	
$(EXEC): %.exe: input.data ref.data %.o main.cc
	$(CC) main.cc $(basename $@).o -o $@ -lm -O3 $(MACROS) $(IFLAGS)

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf $(DFG_HEADERS)

clean: cleandata
	rm -f *.o *.log *.exe
	rm -rf stats gams m5out viz verif

ultraclean: clean cleandfg
	rm  -f *.csv
