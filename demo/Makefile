.PRECIOUS: %.bc

RV_ROOT = $(RISCV)/sysroot/usr
MARCH = rv64imac
ABI = lp64
INCLUDE = -I$(SS_TOOLS)/include -I$(SS)/ss-workloads/common/include
SBCONFIG ?= $(SS)/chipyard/generators/dsagen2/adg/Mesh7x5-Simple64-Full7I5O.json
DEBUG_FLAG = 
CLANG_DEBUG_FLAG=

ifdef DEBUG_SS
DEBUG_FLAG += -debug-only=stream-specialize
CLANG_DEBUG_FLAG += -v
endif


SS_OPTS = $(CLANG_DEBUG_FLAG) -O1        \
          -fno-inline                    \
          -fno-vectorize                 \
	  -fno-slp-vectorize             \
	  -fno-unroll-loops              \
	  -stream-specialize             \
	  -mllvm -no-loop-load-elim=1    \
	  -mllvm -disable-loop-idiom-all \
	  -emit-llvm -c -S
EMU_OPTS = -O3

PREFIX = "$(SS_TOOLS)/bin"

ifneq ($(GEM5), 0)
MACROS += -DGEM5
else
MACROS += -DCHIPYARD
endif

ifdef LONG_TEST
MACROS += -DLONG_TEST
endif

ifdef FAKE
MACROS += -DFAKE=1
endif

ifdef NUM_CORES
# MACROS += -DNUM_CORES=1
MACROS += -DNUM_CORES=$(NUM_CORES)
endif

# Find all the C source code
CC_SRC = $(wildcard *.c)
# Translate C source code into LLVM IR
RAW_LL = $(CC_SRC:.c=.ll)
# Compile the vanilla C code into RISCV executable binaries
OPT = $(CC_SRC:.c=.opt)
# Compile the vanilla C code into host executable binaries
HOST_EXEC = $(addprefix host-, $(CC_SRC:.c=.out))
# Transform the LLVM IR into decoupled-spatial IR
XFORM_LL = $(addprefix ss-,$(RAW_LL))
# ASM generation for decoupled-spatial IR
ASM = $(XFORM_LL:.ll=.s)
# Object binary generation for ASM
OBJ = $(ASM:.s=.o)
# Link the objects into executable binaries
EXEC = $(ASM:.s=.out)
# Link the objects into executable binaries for the purpose of FPGA simulation
CHIPYARD_EXEC = $(ASM:.s=.riscv)
# The cross-compilation compiler
RV_CC ?= riscv64-unknown-elf-gcc

.PRECIOUS: $(RAW_LL)
$(RAW_LL): %.ll: %.c
	clang $< $(SS_OPTS) $(MACROS) $(INCLUDE) --sysroot=$(RV_ROOT)

.PRECIOUS: $(XFORM_LL)
$(XFORM_LL): ss-%.ll: %.ll
	SBCONFIG=$(SBCONFIG) \
	opt -load $(SS)/dsa-llvm-project/build/lib/DSAPass.so -stream-specialize $(DEBUG_FLAG) $< -o $@ -S

.PRECIOUS: $(ASM)
$(ASM): %.s: %.ll
	$(PREFIX)/llc -mcpu=rocket-rv64 -mattr=+m,+d,+f,+a --code-model=medium $<

$(OBJ): %.o: %.s
	riscv64-unknown-linux-gnu-gcc $< -o $@ -lm -fno-common -c

$(EXEC): %.out: %.o common/gem5-harness.c
	riscv64-unknown-linux-gnu-gcc $(MACROS) \
	  common/gem5-harness.c -c -O3 -o gem5-harness.o -lm -fno-common -c
	riscv64-unknown-linux-gnu-gcc $< gem5-harness.o -o $@ -lm -lpthread -fno-common -static

$(CHIPYARD_EXEC): %.riscv: %.o common/chipyard-harness.c
	riscv64-unknown-elf-gcc common/chipyard-harness.c -c -DPREALLOCATE=1 -mcmodel=medany    \
	          -std=gnu99 -O3 -ffast-math -fno-common -fno-builtin-printf \
	          -fno-tree-loop-distribute-patterns -DCHIPYARD $(MACROS)
	riscv64-unknown-elf-gcc $< chipyard-harness.o common/crt.S -o $@ -nostdlib \
	          -nostartfiles -fno-zero-initialized-in-bss -lm -lgcc -T./common/htif.ld $(MACROS)

clean-dsa:
	rm -f ss-*.*
	rm -f *.dfg *.dfg.h

clean: clean-dsa
	rm -rf *.bc *.ll *.o *.out *.s *.dfg *.dfg.h *.bits.h *.log *.riscv sched .extracted-llvm-flags
	rm -rf .sched
	rm -rf viz verif
	rm -f common/*.o
	rm -f *.opt

fake:
	mkdir -p sched
	mkdir -p verif
	mkdir -p viz
	mkdir -p viz/iter
	mkdir -p m5out
	mkdir -p stats
	mkdir -p gams

ultraclean: clean
	rm -rf verif
	rm -rf viz
	rm -rf m5out
	rm -rf stats
	rm -rf gams
	rm -rf dfgs
	rm -f dfgs.list


