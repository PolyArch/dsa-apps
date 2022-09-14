#!/bin/sh

failed=""

export BACKCGRA=1
if [ -z $L1DSIZE ]; then
  export L1DSIZE=32kB
fi
if [ -z $L1DASOC ]; then
  export L1DASOC=8
fi
if [ -z $L2SIZE ]; then
  export L2SIZE=512kB
fi
if [ -z $NUM_CORES ]; then
  export NUM_CORES=4
fi

export L1ISIZE=16kB

if [ -z "$SBCONFIG" ]; then
  export SBCONFIG=${SS}/chipyard/generators/dsagen2/adg/Mesh7x5-Full64-FixFloatSIMD-Full7I5O.json
fi

if [ -z "$DEBUG_SS" ]; then
  GEM5="gem5.opt"
else
  GEM5="gem5.debug"
fi

timeout 120 make $1

FU_FIFO_LEN=15 ${GEM5} $SS/dsa-gem5/configs/example/se.py \
  --cpu-type=MinorCPU --l1d_size=$L1DSIZE --l1d_assoc=$L1DASOC \
  --l1i_size=$L1ISIZE --caches \
  --l2_size=$L2SIZE --l2cache  \
  --num-cpus=$NUM_CORES --cpu-clock=1GHz  --sys-clock=1GHz \
  --mem-type="DDR4_2400_16x4" \
  --cmd=./$1 --options=${OPTIONS}
