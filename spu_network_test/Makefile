TESTS=cpu
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.c
	 $(CPP) $< $(LIB) $(CFLAGS) -static -o $@

run:
	gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=4 --num-dirs=4 --network=simple --topology=Mesh_XY --mesh-rows=2 --cmd=cpu

clean:
	rm -f $(TESTS) *.s *.o


# gem5.opt $(SS_STACK)/gem5/configs/example/se.py --cpu-type=MinorCPU --ruby --num-cpus=4 --num-dirs=4 --network=simple --topology=Mesh_XY --mesh-rows=2 --vecs-per-net=1 --routing-algorithm=1 --cmd=cpu
# $(SS_STACK)/gem5/build/RISCV/gem5.opt ~/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu


