#!/bin/bash
export SBCONFIG=$SS_TOOLS/configs/diannao_simd64.sbmodel


#export LD_LIBRARY_PATH=~/ss-stack/ss_tools/lib

make -j8

tests_passed=0
tests_failed=0
tests_total=0

fail_list=""
pass_list=""

function run_test {
  test=$1
  timeout 10 gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=64kB --l1i_size=16kB --caches  --cmd=$test
  
  if [ "$?" != "0" ]; then
    echo $test FAILED
    tests_failed=$((tests_failed+1))
    fail_list="$fail_list $test"
  else
    tests_passed=$((tests_passed+1))
    pass_list="$pass_list $test"
  fi
  tests_total=$((tests_total+1))
}

if [ -z "$1" ]; then

  echo "Test all the cases!"
  #for i in `ls ind*.c | grep -v fix | grep -v unalign`; do
  for i in `ls *.c | grep -v fix | grep -v unalign`; do
    for l in 4 12 32 36 2048; do
      test=bin/`basename $i .c`_$l
      run_test $test
    done
  done

  for i in `ls fix*.c`; do
    test=bin/`basename $i .c`
    run_test $test
  done

else

  echo "Test " $1
  run_test $1

fi



if [ "$tests_failed" = "0" ]; then
  echo All $tests_passed Tests Passed!
else
  echo $tests_failed / $tests_total Tests FAILED!
  echo "Passing:  $pass_list"
  echo "Failing:  $fail_list"
fi


