ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif
ifndef N
N=32
endif


DFG=compute.dfg writeback.dfg
DFG_HEADERS=$(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

IFLAGS=-I../../common/include

CC=riscv64-unknown-elf-g++

input.data ref.data:
	python gen.py $(N)

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)
MACROS=-D N=$(N) -D VEC_WIDTH=$(VEC_WIDTH)

softbrain.o: softbrain.cc $(DFG_HEADERS)
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS)

$(OBJS): %.o: %.cc
	$(CC) $< -o $@ -c $(MACROS) $(IFLAGS) -O2

$(LOGS): %.log: %
	SBCONFIG=$(SBCONFIG) \
	spike --ic=64:4:64 --dc=64:4:64 --l2=1024:8:64 --extension=softbrain $(SS_TOOLS)/riscv64-unknown-elf/bin/pk $< \
	| tee $@
	
$(BENCHMARKS): %: input.data ref.data %.o main.cc
	$(CC) main.cc $@.o -o $@ -lm -O3 $(MACROS) $(IFLAGS)

$(BENCHMARKS:=clean):
	cd $(@:clean=); make clean

cleandata:
	rm -f input.data ref.data

cleandfg:
	rm -rf verif $(DFG_HEADERS)

clean: cleandfg
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS)
	rm -rf stats gams viz

ultraclean: clean cleandata
