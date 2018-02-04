#!/bin/bash
make -j8

export SBCONFIG=$SS_TOOLS/configs/revel.sbmodel

test_failed=0
for i in *.c; do
  test=`basename $i .c`
  #SUPRESS_SB_STATS=1 spike --extension=softbrain $SS_TOOLS/riscv64-unknown-elf/bin/pk $test
  SUPRESS_SB_STATS=1 gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --caches  --cmd=bin/$test
  if [ "$?" != "0" ]; then
    echo $test FAILED
    test_failed=$((test_failed +1))
  fi
done

if [ "$test_failed" = "0" ]; then
  echo All Tests Passed!
else
  echo Tests Failed: $test_failed
fi


