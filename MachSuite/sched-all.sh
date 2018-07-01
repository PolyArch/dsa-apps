#!/bin/bash

cur_dir=`pwd`
for i in ./gemm/blocked ./md/knn ./spmv/ellpack ./stencil/stencil2d ./stencil/stencil3d; do
echo $i
cd $i
./sched.py
cd $cur_dir
done

