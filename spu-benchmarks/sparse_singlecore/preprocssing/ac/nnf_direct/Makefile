ifndef N_PART
# N_PART=(512+256)/2=768/2=384
N_PART=64
endif

TESTS=part_nnf
TESTC=$(TESTS:=.cpp)


all: $(TESTS)

MACROS= -Ddataset=$(dataset) -DN_PART=$(N_PART)
# MACROS= -DN_PART=$(N_PART)

GCC=g++

process_nnf: process_nnf.cpp
	$(GCC) $< $(MACROS) -o $(@)

circuit.data: process_nnf
	./process_nnf

$(TESTS): $(TESTC) circuit.data
	$(GCC) $< $(MACROS) -o $(@)

run:
	./part_nnf

clean:
	rm part_nnf process_nnf
