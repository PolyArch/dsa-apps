ifndef SS_TOOLS
$(error SS_TOOLS is undefined)
endif

KERN=md
KERN_R5=$(KERN)_r5
KERN_SB=$(KERN)_sb
ALG=knn

CFLAGS?=-O3 -Wall -Wno-unused-label

SRCS=$(KERN).c local_support.c ../../common/support.c
FILES=$(SRCS) $(KERN).h ../../common/support.h

##SB_Related
SB_SRCS=$(KERN_SB).c local_support.c ../../common/support.c
SB_FILES=$(SB_SRCS) $(KERN).h ../../common/support.h

SB_CPP=riscv64-unknown-elf-g++
#SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64_half.sbmodel
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel

DFGS=md_sb.dfg
DFG_HEADERS=$(DFGS:.dfg=.dfg.h)

$(DFG_HEADERS): %.dfg.h: %.dfg
	$(SS_TOOLS)/bin/ss_sched $(SBCONFIG) $<

$(KERN_SB): $(SB_FILES) ../../common/harness.c $(DFG_HEADERS) 
	$(SB_CPP) $(CFLAGS) -I../../common -I$(SS_TOOLS)/include -o $(KERN_SB) $(SB_SRCS) ../../common/harness.c -lm -I ../../../common/include

sim.log: $(KERN_SB) input.data check.data
	SBCONFIG=$(SBCONFIG) BACKCGRA=1 FU_FIFO_LEN=15 \
	gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=MinorCPU \
	--l1d_size=2048kB --l1d_assoc=32 --l1i_size=16kB --l2_size=1024kB --caches \
	--cmd=./$(KERN_SB) --options="input.data check.data" | tee $@

run_r5: $(KERN_R5) input.data check.data
	SBCONFIG=$(SB_CONFIG) spike  --ic=128:4:64 --dc=128:4:64 --l2=1024:8:64  --extension=softbrain $(SS_TOOLS)/riscv64-unknown-elf/bin/pk ./$(KERN_R5) input.data check.data


$(KERN_R5): $(FILES) ../../common/harness.c
	$(SB_CPP) $(CFLAGS) -I../../common -o $(KERN_R5) $(SRCS) ../../common/harness.c

$(KERN): $(FILES) ../../common/harness.c
	$(CC) $(CFLAGS) -I../../common -o $(KERN) $(SRCS) ../../common/harness.c

run: $(KERN) input.data check.data
	./$(KERN) input.data check.data

generate: $(FILES) generate.c
	$(CC) $(CFLAGS) -I../../common -o generate $(SRCS) generate.c
	./generate

hls: $(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(KERN_SB) *.dfg.h
