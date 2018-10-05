ifndef C
C=4
endif

TESTS=main
TESTC=$(TESTS:=.cpp)
TEST_OBJS=$(TESTS:=.o)

all: $(TESTS)

CC=riscv64-unknown-linux-gnu-gcc
CPP=riscv64-unknown-linux-gnu-g++

OPT?=-O0 #-Wall
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl

MACROS= -DC=$(C)

# targets is main.o and dependent on main.cpp
$(TEST_OBJS): $(TESTC)
	 $(CC) -c $< $(CFLAGS) $(MACROS) -I. -static -o $(@)
	
# targets is main and dependent on main.o
$(TESTS): $(TEST_OBJS)
	 $(CPP) $(LIB) $(CFLAGS) -static -o $(@) $<

run:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=16kB --l1i_size=16kB --l2_size=64kB --caches --num-cpus=4 --cmd=$(TESTS)

run-classic:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=16kB --l1i_size=16kB --l2_size=64kB --caches --num-cpus=4 --cmd=$(TESTS)

run-ruby:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=4 --num-dirs=4 --network=simple --topology=Mesh_XY --mesh-rows=2 --cmd=$(TESTS)

clean:
	rm -f $(TESTS) *.s *.o
