ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

ifndef N
N=64
endif

ifndef VEC_WIDTH
VEC_WIDTH=2
endif

ifeq ($(VEC_WIDTH), 2)
DFG_DIR=sb-2-wide
else
ifeq ($(VEC_WIDTH), 4)
DFG_DIR=sb-4-wide
else
$(error Unsupported vectorization width! Abort!)
endif
endif

DFG=$(DFG_DIR)/compute.dfg
DFG_HEADERS=$(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I$(DFG_DIR)
CFLAGS=-O3
LFLAGS=-lm

CC=riscv64-unknown-elf-g++

input.data ref.data:
	python gen.py $(N)

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)
MACROS=-D N=$(N) -D VEC_WIDTH=$(VEC_WIDTH)

softbrain.o: softbrain.c $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) $(CFLAGS) $(LFLAGS)

$(OBJS): %.o: %.c
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) $(CFLAGS) $(LFLAGS)

$(LOGS): %.log: %
	SBCONFIG=$(SBCONFIG) \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=./$< \
	| tee $@
	
$(BENCHMARKS): %: input.data ref.data %.o main.c
	$(CC) main.c $@.o -o $@ $(CFLAGS) $(LFLAGS) $(MACROS) $(IFLAGS)

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf sb-2-wide/compute.h sb-2-wide/verif sb-2-wide/viz
	rm -rf sb-4-wide/compute.h sb-4-wide/verif sb-4-wide/viz

clean: cleandfg
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS)
	rm -rf stats gams viz m5out

ultraclean: clean cleandata
