ifdef N
MACROS = -DN=$(N)
endif

ifdef U
MACROS += -DU=$(U)
endif

ifdef U1
MACROS += -DU1=$(U1)
endif

ifdef U2
MACROS += -DU2=$(U2)
endif

ifdef U3
MACROS += -DU3=$(U3)
endif

ifdef U4
MACROS += -DU4=$(U4)
endif

ifdef U5
MACROS += -DU5=$(U5)
endif

IND ?= 1

ifeq ($(IND), 0)
MACROS += -DNIND
endif

ifdef DTYPE
MACROS += -DDTYPE=$(DTYPE)
endif

MACROS += -DGEM5

include ../Common/Makefile.inc
