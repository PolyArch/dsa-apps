ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS)/ss-scheduler/configs/revel-1x2.sbmodel
endif

TESTN=$(wildcard fix*.c)
TESTALL=$(wildcard *.c)
TESTC=$(filter-out $(TESTN), $(TESTALL))

TESTSN=$(patsubst %.c,bin/%,$(TESTN))
TESTS_SMALL=$(patsubst %.c,bin/%_36,$(TESTC))
TESTS_MEDIUM=$(patsubst %.c,bin/%_128,$(TESTC))
TESTS_LARGE=$(patsubst %.c,bin/%_2048,$(TESTC))

TESTS=$(TESTS_SMALL) $(TESTS_MEDIUM) $(TESTS_LARGE) $(TESTSN) 

all: bin $(TESTS) 

# CC=riscv64-unknown-linux-gnu-gcc
CC=riscv64-unknown-linux-gnu-g++
CPP=riscv64-unknown-linux-gnu-g++

DFGS = $(wildcard *.dfg)
DFG_HEADERS = $(DFGS:.dfg=.dfg.h)
HEADERS = testing.h check.h
IFLAGS = -I$(SS)/ss-tools/include/ss-intrin
#${info $(DFG_HEADERS)}

bin:
	mkdir -p bin

# It should not be executed if no dfg is changed?
$(DFG_HEADERS): %.dfg.h: %.dfg
	$(RISCV)/bin/ss_sched -a sa $(SBCONFIG) $<

OPT?=-O3
CFLAGS=$(OPT) -g -ggdb -gdwarf-3 -static -DOPENMP -fopenmp
LIB=-L. -lgomp -lpthread -ldl
LFLAGS = -lm

# $(TEST_OBJS): bin/% : %.c $(DFG_HEADERS) $(HEADERS) bin
# 	 $(CC) -c $< $(CFLAGS) $(MACROS) $(IFLAGS) -I. -static -o $(@) -DASIZE=16
# 	
# # $(TESTSN): bin/% : $(TEST_OBJS)
# $(TESTSN): $(TEST_OBJS) $(DFG_HEADERS) $(HEADERS) bin
# 	 $(CPP) $< $(LIB) $(CFLAGS) $(IFLAGS) -static -o $@ -DASIZE=23
	 
$(TESTSN): bin/% : %.c $(DFG_HEADERS) $(HEADERS) bin
	$(CC) -c $< $(CFLAGS) $(MACROS) $(IFLAGS) -I. -static -o main.o -DASIZE=16
	$(CPP) $(LIB) $(CFLAGS) $(IFLAGS) -static -o $@ main.o -DASIZE=16


$(TEST_OBJS_SMALL): %.c $(DFG_HEADERS) $(HEADERS) bin
	 $(CC) -c $< $(CFLAGS) $(MACROS) $(IFLAGS) -I. -static -o $(@) -DASIZE=12
	
$(TESTS_SMALL): bin/%_36 : $(TEST_OBJS_SMALL)
	 $(CPP) $(LIB) $(CFLAGS) $(IFLAGS) -static -o $@ $< -DASIZE=12

$(TEST_OBJS_MEDIUM): %.c $(DFG_HEADERS) $(HEADERS) bin
	 $(CC) -c $< $(CFLAGS) $(MACROS) $(IFLAGS) -I. -static -o $(@) -DASIZE=500
	
$(TESTS_MEDIUM): bin/%_500 : $(TEST_OBJS_MEDIUM)
	 $(CPP) $(LIB) $(CFLAGS) $(IFLAGS) -static -o $@ $< -DASIZE=500

$(TEST_OBJS_LARGE): %.c $(DFG_HEADERS) $(HEADERS) bin
	 $(CC) -c $< $(CFLAGS) $(MACROS) $(IFLAGS) -I. -static -o $(@) -DASIZE=2048
	
$(TESTS_LARGE): bin/%_2048 : $(TEST_OBJS_LARGE)
	$(CPP) $(LIB) $(CFLAGS) $(IFLAGS) -static -o $@ $< -DASIZE=2048

clean:
	rm -f $(TESTS) $(DFG_HEADERS) *.s *.o
	rm -rf m5out viz verif stats gams bin
