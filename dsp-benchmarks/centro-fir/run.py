#!/usr/bin/env python3
import imp, os

run = imp.load_source('run', '../tools/run.py')

SS = os.getenv('SS')

run.run([37, 199], 'M=%d ', ['origin', 'new', 'latency'], 'fir.res')
run.run([37, 199], 'M=%d ', ['ds'], 'fir.res')
run.run([(164, 37), (326, 199)],
        'SBCONFIG=%s/ss-scheduler/configs/revel-4x4.sbmodel N=%%d M=%%d ' % SS, ['ds'], 'fir.res')
