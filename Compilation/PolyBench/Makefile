ifdef DATA_SIZE
  MACROS = -DNI=$(DATA_SIZE) -DNJ=$(DATA_SIZE) -DNK=$(DATA_SIZE) -DNL=$(DATA_SIZE) -DNM=$(DATA_SIZE) -DN=$(DATA_SIZE)
else
  ifndef DEBUG_SS
    MACROS = -DNI=128 -DNJ=128 -DNK=128 -DNL=128 -DNM=128 -DN=128
  else
    MACROS = -DNI=32 -DNJ=32 -DNK=32 -DNL=32 -DNM=32 -DN=32
  endif
endif

ifdef U
  MACROS += -DU=$U
else
  MACROS += -DU=4
endif

ifdef U1
  MACROS += -DU1=$(U1)
endif

ifdef U2
  MACROS += -DU2=$(U2)
endif

include ../Common/Makefile.inc
