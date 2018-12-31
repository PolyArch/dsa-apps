TESTS=encode_act

TESTS2=encode_weights

all: $(TESTS) $(TESTS2)

CPP=g++
# MACROS= -Dact_file=$(act_file) -Dwgt_file=$(wgt_file) -DNi=$(Ni) -DNx=$(Nx) -DNy=$(Ny) -DTx=$(Tx) -DTy=$(Ty) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DKx=$(Kx) -DKy=$(Ky)
MACROS= -Dnet_name=$(net_name) -DNi=$(Ni) -DNx=$(Nx) -DNy=$(Ny) -DTx=$(Tx) -DTy=$(Ty) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DKx=$(Kx) -DKy=$(Ky)

# compile 
$(TESTS): encode_act.cpp
	 $(CPP) $< $(MACROS) -o $(@)

$(TESTS2): encode_weights.cpp
	 $(CPP) $< $(MACROS) -o $(@)


run:
	./encode_act
	./encode_weights

clean:
	rm encode_act encode_weights
