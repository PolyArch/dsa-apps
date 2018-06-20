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
CFLAGS = -std=c++0x -O2 -Wall
MACROS = -DN=$(N) -DM=$(M) -DS0=$(S0) -DS1=$(S1)

main.exe: main.cc
	$(CC) $(CFLAGS) $(MACROS) $< -o $@
