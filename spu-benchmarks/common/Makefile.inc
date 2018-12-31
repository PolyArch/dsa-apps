ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/spu_merge_test.sbmodel
endif

TESTS=main
TESTC=$(TESTS:=.cpp)
TEST_OBJS=$(TESTS:=.o)

all: $(TESTS)

CC=riscv64-unknown-linux-gnu-g++
CPP=riscv64-unknown-linux-gnu-g++

OPT?=-O0 #-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3  -DOPENMP -fopenmp -I$(SS)/ss-workloads/common/include -I$(SS_TOOLS)/include # -I$(SS_TOOLS)/include/ss-intrin
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

cleandfg:
	rm -f *.dfg.h
	rm -r sched

cleandata:
	rm -f *.data

cleancpu:
	rm -f cpu

clean:
	rm -f $(TESTS) *.s *.o
	rm -r m5out stats verif viz gams

ultraclean: cleandata cleancpu clean
	rm -f *.log
