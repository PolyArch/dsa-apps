ifndef SBCONFIG
$(warning SBCONFIG is undefined, using default)
SBCONFIG=$(SS_TOOLS)/configs/diannao_simd64.sbmodel
endif

ifndef N
N=1024
endif

ifndef M
M=191
endif

DFG = compute.dfg

DFG_HEADERS=$(DFG:.dfg=.h)

SCALE =  $(N) $(M)

MACROS=-D N=$(N) -D M=$(M)

include ../common/Makefile.inc
