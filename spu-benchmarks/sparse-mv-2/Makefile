ifndef N
N=16
endif

ifndef M
M=16
endif

ifndef S0
S0=0.64
endif

ifndef S1
S1=0.66
endif

ifndef SBCONFIG
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

CC = g++
RCC= riscv64-unknown-elf-g++
CFLAGS = -std=c++0x -O3 -g
LFLAGS = -lm
MKL_IFLAGS = -I../../common/include -I$(SS_TOOLS)/include -I ~/intel/mkl/include 
MKL_LFLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl  -L"~/intel/mkl/lib"
NUM_THREADS = 8
MKL_MACROS = -DNUM_PTHREADS=$(NUM_THREADS) -DOMP_NUM_THREADS=$(NUM_THREADS) -DNUM_OMP_THREADS=$(NUM_THREADS)
#CFLAGS = -std=c++0x
MACROS = -DN=$(N) -DM=$(M) -DS0=$(S0) -DS1=$(S1)

ifdef DEBUG
MACROS += -DDEBUG
endif

DFG = eie.dfg
DFG_HEADERS = $(DFG:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/sb_sched --max-iters=100000 --verbose $(SBCONFIG) $< 

input.data output.data: gen.py
	python $< $(N) $(M) $(S0) $(S1)

main.exe: main.cc input.data output.data
	$(CC) $(CFLAGS) $(MACROS) $< -o $@ $(LFLAGS)

mkl.exe: mkl.cc
	$(CC) $< -o $@ $(LFLAGS) $(MKL_LFLAGS) $(CFLAGS) $(MACROS) $(MKL_MACROS) $(MKL_IFLAGS)

softbrain.exe: main.cc input.data output.data $(DFG_HEADERS)
	$(RCC) $(CFLAGS) $(MACROS) $< -o $@ $(LFLAGS)

cleandata:
	rm -f input.data output.data

cleandfg:
	rm -f *.dfg.h

clean:
	rm -f *.exe

ultraclean: cleandata clean cleandfg
	rm -f *.log


