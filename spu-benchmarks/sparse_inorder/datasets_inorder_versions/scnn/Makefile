ifndef Tx
Tx=32
endif

ifndef Ty
Ty=32
endif

ifndef Ky
Ky=32
endif

ifndef Kx
Kx=3
endif

ifndef Ni
Ni=64
endif

ifndef core_per_spad
core_per_spad=4
endif

ifndef Tn
Tn=1
endif

ifndef neuron_sp
neuron_sp=0.25
endif

ifndef synapse_sp
synapse_sp=0.45
endif

ifndef rle_width
rle_width=4
endif

TESTS=cpu
TESTC=$(TESTS:=.cpp)

all: $(TESTS)

CPP=riscv64-unknown-elf-g++
# MACROS= -DTx=$(Tx) -DNi=$(Ni) -DTn=$(Tn) -Dneuron_sp=$(neuron_sp) -Dsynapse_sp=$(synapse_sp) -Drle_width=$(rle_width) -Dcore_per_spad=$(core_per_spad) -DTy=$(Ty) -DKy=$(Ky)

MACROS= -Dnet_name=$(net_name) -DNi=$(Ni) -DNx=$(Nx) -DNy=$(Ny) -DTx=$(Tx) -DTy=$(Ty) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DKx=$(Kx) -DKy=$(Ky)

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 
#--std=c++11 
LFLAGS = -lm

$(TESTS): % :%.cpp
	 $(CPP) $< $(LIB) $(CFLAGS) $(MACROS) -static -o $@ $(LFLAGS)

run:
	gem5.opt /home/vidushi/ss-stack/gem5/configs/example/se.py --cpu-type=MinorCPU --l1d_size=64kB --l1i_size=16kB --l2_size=1024kB --caches --cmd=cpu


cleandata:
	rm -f input_neuron.data input_synapse.data

clean:
	rm -f $(TESTS)  *.s *.o

ultraclean: cleandata cleancpu clean
	rm -f *.log
