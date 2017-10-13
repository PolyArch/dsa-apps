ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

ifndef N
N=4
endif

DFG = compute.dfg
DFG_HEADERS=$(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I ../../../ss-tools/include
CFLAGS=-O3
LFLAGS=-lm

CC=riscv64-unknown-elf-g++

input.data ref.data:
	python gen.py $(N)

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)
MACROS=-D N=$(N) -D VEC_WIDTH=$(VEC_WIDTH)

softbrain.o: softbrain.cc $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) $(CFLAGS) $(LFLAGS)

$(OBJS): %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) $(CFLAGS) $(LFLAGS)

$(LOGS): %.log: %
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@
	
$(BENCHMARKS): %: input.data ref.data %.o main.cc
	$(CC) main.cc $@.o -o $@ $(CFLAGS) $(LFLAGS) $(MACROS) $(IFLAGS)

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf sb-8-wide/compute.h sb-8-wide/verif sb-8-wide/viz

clean: cleandfg
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS) $(DFG_HEADERS)
	rm -rf stats gams viz verif m5out

ultraclean: clean cleandata
