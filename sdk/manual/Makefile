ifndef ADG
$(warning ADG is undefined, using default)
ADG = ../../../chipyard/generators/dsagen2/adg/Mesh7x5-Full64-SIMD-Full7I5O.json
endif

IFLAGS = -I$(SS_TOOLS)/include/

%.dfg.h: %.dfg
	ss_sched -v $(ADG) $<

%.dfg.bits.h: %.dfg
	ss_sched -b -v $(ADG) $<

CC = riscv64-unknown-linux-gnu-g++

%.out: %.c add.dfg.h
	$(CC) -DNUM_CORES=$(NUM_CORES) -DGEM5 -O3 -std=c++11 $(IFLAGS) $(MACROS) -o $@ $< -static


%.riscv: main.c compute.dfg.bits.h
	riscv64-unknown-linux-gnu-gcc ../Common/chipyard-harness.c -c -DPREALLOCATE=1 \
	  -mcmodel=medany -std=gnu99 -O3 -ffast-math -fno-common -fno-builtin-printf  \
	  -fno-tree-loop-distribute-patterns -DCHIPYARD -DNUM_CORES=$(NUM_CORES)
	$(CC) -DNUM_CORES=$(NUM_CORES) -DCHIPYARD -O3 -std=c++11 $(IFLAGS) -o $@ $<           \
	  -DPREALLOCATE=1 -mcmodel=medany -O3 -ffast-math -fno-common -fno-builtin-printf          \
	  -nostdlib -fno-tree-loop-distribute-patterns -lm -lgcc -nostartfiles -T../Common/htif.ld \
	  ../Common/crt.S chipyard-harness.o -static

clean:
	rm -rf *.dfg.h *.dfg.bits.h *.out m5out sched stats verif viz *.riscv *.o

ultraclean: clean
	rm -rf verif
	rm -rf viz
	rm -rf m5out
	rm -rf stats
	rm -rf gams
