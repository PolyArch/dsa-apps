all: a.out

MACROS=-Ddataset=$(dataset) -DN=$(N) -DM=$(M) -Dfeat_needed=$(feat_needed)

a.out: bin_dataset.cpp
	g++ $< $(MACROS) -o $(@)

run:
	./a.out

clean:
	rm a.out
