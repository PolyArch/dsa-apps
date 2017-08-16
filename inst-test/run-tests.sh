#!/bin/bash
make -j8

export SBCONFIG=$SS_TOOLS/configs/diannao_simd64.sbmodel

test_failed=0
for i in *.c; do
  test=`basename $i .c`
  SUPRESS_SB_STATS=1 spike --extension=softbrain $SS_TOOLS/riscv64-unknown-elf/bin/pk $test
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


