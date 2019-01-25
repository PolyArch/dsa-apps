TESTS=cpu
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++
MACROS= -DN=$(N) -DM=$(M) -Dlayer_name=$(layer_name)
GCC=g++

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

$(TESTS): % :%.cpp
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

run:
	gem5.opt /home/vidushi/ss-stack/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu

cleandata:
	rm -f *.data

clean:
	rm -f $(TESTS) *.s *.o

ultraclean: cleandata cleancpu clean
	rm -f *.log
