ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

DFG_HEADERS = $(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --verbose $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I ../../../ss-tools/include -I ../common/ -I ./dfgs

CC=riscv64-unknown-elf-g++

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)

input.data ref.data:
	python gen.py $(SCALE)

softbrain.o: softbrain.cc $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS)

$(OBJS): %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O2

$(LOGS): %.log: %
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@
	
$(BENCHMARKS): %: input.data ref.data %.o main.cc
	$(CC) main.cc $@.o -o $@ -lm -O3 $(MACROS) $(IFLAGS)

softbrain-all:
	for i in $(CASES) ; do \
		make clean > /dev/null 2>&1; \
		N=$$i make softbrain.log > /dev/null 2>&1; \
		echo $$i; \
		cat softbrain.log | grep Cycles; \
		cat softbrain.log | grep -A2 ACCEL; \
	done

optimized-all:
	for i in $(CASES) ; do \
		make clean > /dev/null 2>&1; \
		N=$$i make optimized.log > /dev/null 2>&1; \
		echo $$i; \
		cat optimized.log | grep Cycles; \
	done

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf $(DFG_HEADERS)

clean: cleandata
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS)
	rm -rf stats gams m5out viz verif

ultraclean: clean cleandfg
