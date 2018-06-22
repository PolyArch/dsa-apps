ifndef N
N=128
endif

ifndef M
M=128
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
CFLAGS = -std=c++0x -O3
LFLAGS = -lm
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


