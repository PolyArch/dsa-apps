ifndef N
N=4096
endif

ifndef M
M=1024
endif

ifndef S0
S0=0.1
endif

ifndef S1
S1=1
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

input.data output.data: gen.py
	python $< $(N) $(M) $(S0) $(S1)

main.exe: main.cc input.data output.data
	$(CC) $(CFLAGS) $(MACROS) $< -o $@ $(LFLAGS)

cleandata:
	rm -f input.data output.data

clean:
	rm -f main.exe

ultraclean: cleandata clean
