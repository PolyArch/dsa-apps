#!/bin/bash

declare -a arr=("cholesky" "qr" "qr2" "svd" "centro-fir" "gemm" "fft")

for i in "${arr[@]}"
do
	echo $i
	cd ./$i
	./run.py
	cd ..
done
