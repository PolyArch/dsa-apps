#!/bin/bash

declare -a arr=("centro-fir" "gemm" "fft" "cholesky")

for i in "${arr[@]}"
do
	echo $i
	cd ./$i
	./sched.py
	cd ..
done
