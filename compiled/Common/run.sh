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

export L1ISIZE=16kB

if [ -z "$SBCONFIG" ]; then
  export SBCONFIG=${SS}/chipyard/generators/dsagen2/adg/Mesh7x5-Full64-FixFloatSIMD-Full7I5O.json
fi

if [ -z "$DEBUG_SS" ]; then
  GEM5="gem5.opt"
else
  GEM5="gem5.debug"
fi

if [ -z "$NUM_CORES" ]; then
  NUM_CORES=1
fi

if [ -z "$COMPAT_ADG" ]; then
  COMPAT_ADG=0
fi

if [ -z $1 ]; then

  make clean
  for i in `ls *.c`
  do
    out=ss-${i%.c}.out
    # make clean
    timeout 120 make $out

    if [ $? -eq 0 ]; then
      echo Compilation done...
    else
      echo $out Complation FAILED!
      failed="${failed}"$'\n'"${out}(compilation)"
      continue
    fi

    FU_FIFO_LEN=15 ${GEM5} $SS/dsa-gem5/configs/example/se.py \
      --cpu-type=MinorCPU --l1d_size=$L1DSIZE --l1d_assoc=$L1DASOC \
      --l1i_size=$L1ISIZE --l2_size=$L2SIZE --caches --l2cache \
      --num-cpus=8 --cpu-clock=1GHz  --sys-clock=1GHz \
      --mem-type="DDR4_2400_16x4" \
      --cmd=./$out > $out.log 2>&1
  
    if [ $? -eq 0 ]; then
      echo OK!
    else
      echo $out FAILED!
      failed="${failed}"$'\n'"$out(simulation) "
    fi

  done

  if [ "$failed" != "" ]; then
    echo "Failed: " $failed
    echo $failed > FailList
    exit 1
  else
    echo "All pass! Cong!"
  fi

else

  if [ -z $2 ]; then
    echo $2
  else
    make clean
  fi

  timeout 120 make $1

  FU_FIFO_LEN=15 ${GEM5} $SS/dsa-gem5/configs/example/se.py \
    --cpu-type=MinorCPU --l1d_size=$L1DSIZE --l1d_assoc=$L1DASOC \
    --l1i_size=$L1ISIZE --caches \
    --l2_size=$L2SIZE --l2cache  \
    --num-cpus=$NUM_CORES --cpu-clock=0.09GHz  --sys-clock=1GHz \
    --mem-type="DDR4_2400_16x4" \
    --cmd=./$1 --options=${OPTIONS}

fi
