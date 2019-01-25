TESTS=cpu
TESTC=$(TESTS:=.cc)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++

MACROS= -Ddataset=$(dataset)
# -Dd1=$(d1) -Dd2=$(d2) -Dratio=$(ratio) -DB=$(B) -Dcores_per_spad=$(cores_per_spad) -Dnum_pe=$(num_pe)


OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 

$(TESTS): % :%.cc
	 $(CPP) $< $(MACROS) $(LIB) $(CFLAGS) -static -o $@

run:
	gem5.opt /home/vidushi/ss-stack/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu

cleandata:
	rm -f *.data

clean:
	rm -f $(TESTS) *.s *.o

ultraclean: clean cleandata
