ifndef d1
d1=1
endif

ifndef d2
d2=8
endif

ifndef ratio
ratio=0.5
endif

ifndef B
B=1
endif

ifndef cores_per_spad
cores_per_spad=4
endif

ifndef num_pe
num_pe=64
endif

TESTS=cpu
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

MACROS= -Dd1=$(d1) -Dd2=$(d2) -Dratio=$(ratio) -DB=$(B) -Dcores_per_spad=$(cores_per_spad) -Dnum_pe=$(num_pe)

input.data: complete_gen.py
	python $< $(d1) $(d2) $(ratio) 

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cpp input.data
	 $(CPP) $< $(MACROS) $(LIB) $(CFLAGS) -static -o $@

run:
	/home/vidushi/gem5/build/RISCV/gem5.opt ~/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu

cleandata:
	rm -f *.data

clean:
	rm -f $(TESTS) *.s *.o

ultraclean: clean cleandata
