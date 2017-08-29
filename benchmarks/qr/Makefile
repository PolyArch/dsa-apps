ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

WIDTHS=8

ifndef N
N=32
endif

ifndef VEC_WIDTH
VEC_WIDTH=$(firstword $(WIDTHS))
endif

ifeq ($(findstring $(VEC_WIDTH), $(WIDTHS)),)
$(error Unsupported vectorization width! Abort!)
endif

DFG_DIR=sb-$(VEC_WIDTH)-wide

DFG=$(DFG_DIR)/compute.dfg
DFG_HEADERS=$(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

IFLAGS=-I../../common/include -I$(DFG_DIR)

CC=riscv64-unknown-elf-g++

input.data ref.data:
	python gen.py $(N)

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)
MACROS=-D N=$(N) -D VEC_WIDTH=$(VEC_WIDTH)

$(OBJS): %.o: %.c $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS)

$(LOGS): %.log: %
	SBCONFIG=$(SBCONFIG) \
	spike --ic=64:4:64 --dc=64:4:64 --l2=1024:8:64 --extension=softbrain $(SS_TOOLS)/riscv64-unknown-elf/bin/pk $< \
	| tee $@
	
$(BENCHMARKS): %: input.data ref.data %.o main.c
	$(CC) main.c $@.o -o $@ -lm -O3 $(MACROS) $(IFLAGS)

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf sb-8-wide/compute.h sb-8-wide/verif sb-8-wide/viz

clean: cleandfg
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS)
	rm -rf stats gams viz

ultraclean: clean cleandata
