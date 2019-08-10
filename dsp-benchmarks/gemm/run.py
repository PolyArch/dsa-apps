#!/usr/bin/env python3
import imp, os

run = imp.load_source('run', '../tools/run.py')

SS = os.getenv('SS')

#run.run([12, 48], 'N=%d M=16 P=64 ', ['origin', 'new', 'latency'], log='gemm.res')
run.run([12, 48], 'SBCONFIG=%s/ss-scheduler/configs/revel-4x4.sbmodel N=%%d M=16 P=64 ' % SS, ['ds'],
        'gemm.res')
run.run([2, 6], 'SBCONFIG=%s/ss-scheduler/configs/revel-4x4.sbmodel N=%%d M=16 P=64 ' % SS, ['ds'],
        'gemm.res')

