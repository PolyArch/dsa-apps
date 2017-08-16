#!/bin/bash
export SBCONFIG=$SS_TOOLS/configs/diannao_simd64.sbmodel

make -j8

tests_passed=0
tests_failed=0
tests_total=0

for i in *.c; do
  for l in 4 12 32 36 2048; do
    test=bin/`basename $i .c`_$l
    SUPRESS_SB_STATS=1 spike --extension=softbrain $SS_TOOLS/riscv64-unknown-elf/bin/pk $test
    
    if [ "$?" != "0" ]; then
      echo $test FAILED
      tests_failed=$((tests_failed+1))
    else
      tests_passed=$((tests_passed+1))
    fi
    tests_total=$((tests_total+1))
  done
done

if [ "$tests_failed" = "0" ]; then
  echo All $tests_passed Tests Passed!
else
  echo $tests_failed / $tests_total Tests FAILED!
fi


