#!/usr/bin/env python3
import imp, os

run = imp.load_source('run', '../tools/run.py')

#run.run([64, 1024], 'N=%d ', ['origin', 'new'], 'fft.res')

SS = os.getenv('SS')
#run.run([64, 1024], 'SBCONFIG=%s/ss-scheduler/configs/revel-5x5.sbmodel N=%%d ' % SS, \
#        ['ds'], 'fft.res')
run.run([64, 1024], 'SBCONFIG=%s/ss-scheduler/configs/revel-4x4.sbmodel N=%%d ' % SS, \
        ['ds'], 'fft.res')
