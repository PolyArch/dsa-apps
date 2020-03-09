#!/bin/sh

failed=""

export BACKCGRA=1
export L1DSIZE=2048kB
export L1DASOC=32
export L1ISIZE=16kB
export L2SIZE=2048kB

if [ -z "$SBCONFIG" ]; then
export SBCONFIG="$SS/ss-scheduler/configs/revel-1x1.sbmodel"
fi

if [ -z $1 ]; then

  make clean
  for i in `ls *.cc`
  do
    out=opt-ss-${i%.cc}.out
    make clean
    timeout 120 make $out

    if [ $? -eq 0 ]; then
      echo Compilation done...
    else
      echo $out Complation FAILED!
      failed="${failed} ${out}"
      continue
    fi

    FU_FIFO_LEN=15 gem5.opt $SS/gem5/configs/example/se.py \
      --cpu-type=MinorCPU --l1d_size=$L1DSIZE --l1d_assoc=$L1DASOC \
      --l1i_size=$L1ISIZE --l2_size=$L2SIZE --caches \
      --ruby --num-cpus=8 --num-dirs=8 --network=simple \
      --topology=Mesh_XY --mesh-rows=2 \
      --cmd=./$out
  
    if [ $? -eq 0 ]; then
      echo OK!
    else
      echo $out FAILED!
      failed="${failed} $out"
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

  make clean
  timeout 120 make $1
  FU_FIFO_LEN=15 gem5.opt $SS/gem5/configs/example/se.py \
    --cpu-type=MinorCPU --l1d_size=$L1DSIZE --l1d_assoc=$L1DASOC \
    --l1i_size=$L1ISIZE --l2_size=$L2SIZE --caches \
    --ruby --num-cpus=8 --num-dirs=8 --network=simple \
    --topology=Mesh_XY --mesh-rows=2 \
    --cmd=./$1

fi
