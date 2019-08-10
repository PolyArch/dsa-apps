#!/bin/bash

mkdir -p results

#declare -a arr=("cholesky" "qr" "svd" "centro-fir" "gemm" "fft" "solver")
declare -a arr=("cholesky" "centro-fir" "gemm" "fft" "solver")

for i in "${arr[@]}"
do
	echo $i
	cd ./$i
	./run.py
        mv $i".res" ../results
	cd ..
done
