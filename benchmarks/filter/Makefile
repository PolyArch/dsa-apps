ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif
ifndef N
N=1024
endif

ifndef FILTER
FILTER=512
endif

DFG=compute.dfg
DFG_HEADERS=$(DFG:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched $(SBCONFIG) $< 

IFLAGS=-I../../common/include

CC=riscv64-unknown-elf-g++

input.data ref.data:
	python gen.py $(N) $(FILTER)

BENCHMARKS=baseline optimized softbrain
OBJS=$(BENCHMARKS:=.o)
LOGS=$(BENCHMARKS:=.log)
MACROS=-D N=$(N) -D FILTER=$(FILTER)

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
	rm -rf sb-2-wide/compute.h sb-2-wide/verif sb-2-wide/viz
	rm -rf sb-4-wide/compute.h sb-4-wide/verif sb-4-wide/viz

clean: cleandfg
	rm -f $(OBJS) $(BENCHMARKS) $(LOGS)
	rm -rf stats gams viz

ultraclean: clean cleandata
