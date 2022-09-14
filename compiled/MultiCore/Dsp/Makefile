include ../Common/Makefile.inc

ifndef NUM_CORES
  MACROS += -DNUM_CORES=2
endif

ifdef DATA_SIZE
  MACROS += -DN=$(DATA_SIZE)
endif

ifdef U
  MACROS += -DU=$(U)
endif
