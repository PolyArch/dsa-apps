TESTS=encode_act

TESTS2=encode_weight

all: $(TESTS) $(TESTS2)

CPP=g++
MACROS= -Dnet_name=$(net_name) -DN=$(N) -DM=$(M)

# compile 
$(TESTS): encode_act.cpp
	 $(CPP) $< $(MACROS) -o $(@)

$(TESTS2): encode_weight.cpp
	 $(CPP) $< $(MACROS) -o $(@)


run:
	./encode_act
	./encode_weight

clean:
	rm -f encode_act encode_weight
