.PRECIOUS: %.bc

RV_ROOT = $(SS_TOOLS)/riscv64-unknown-elf
MARCH = rv64imac
ABI = lp64
INCLUDE = -I$(SS_TOOLS)/include \
	  -I$(SS)/ss-workloads/common/include \
	  -I$(RV_ROOT)/include/c++/8.1.0/ \
	  -I$(RV_ROOT)/include/c++/8.1.0/riscv64-unknown-elf/$(ARCH)/$(ABI)/ \
	  -I$(RV_ROOT)/include/c++/8.1.0/riscv64-unknown-elf/

ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS)"/ss-scheduler/configs/revel-1x1.sbmodel"
endif

ifdef DEBUG_SS
DEBUG_FLAG=-debug-only=stream-specialize
CLANG_DEBUG_FLAG=-v
else
DEBUG_FLAG=
CLANG_DEBUG_FLAG=
endif

SS_OPTS = $(CLANG_DEBUG_FLAG) -O2 -fno-inline -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -stream-specialize -emit-llvm -c -std=c++11
OMP_OPTS = $(CLANG_DEBUG_FLAG) -O0 -fno-inline -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fopenmp -emit-llvm -c -std=c++11
EMU_OPTS = $(CLANG_DEBUG_FLAG) -O3 -std=c++11

PREFIX = "$(SS_TOOLS)/bin"

ifndef MACROS
MACROS = 
endif

%.bc: %.cc
	$(PREFIX)/clang++ $< -o $@ $(SS_OPTS) $(MACROS) $(INCLUDE) --sysroot=$(RV_ROOT) 

ss-%.bc: %.bc
	SBCONFIG=$(SBCONFIG) \
	opt $(DEBUG_FLAG) -load $(SS)/dsa-pass/build/libStreamSpecialize.so -stream-specialize $< -o $@

opt-%.bc: %.bc
	$(PREFIX)/opt -O3 < $< > $@

%.ll: %.bc
	$(PREFIX)/llvm-dis < $< > $@

%.s: %.bc
	$(PREFIX)/llc -mattr=+m,+f,+a,+c $<

%.out: %.s
	riscv64-unknown-elf-g++ $< -o $@ -march=$(MARCH) -mabi=$(ABI) -lm

native-%.out: %.cc
	g++ $< -o $@ $(EMU_OPTS) $(MACROS) $(INCLUDE)

clean:
	rm -f *.bc *.ll *.o *.out *.s *.dfg.h *.dfg

fake:
	mkdir -p sched
	mkdir -p verif
	mkdir -p viz
	mkdir -p viz/iter
	mkdir -p m5out
	mkdir -p stats
	mkdir -p gams

ultraclean: clean
	rm -rf sched
	rm -rf verif
	rm -rf viz
	rm -rf m5out
	rm -rf stats
	rm -rf gams
