#!/usr/bin/env python3
import imp
import os

run = imp.load_source('run', '../tools/run.py')

run.run([12, 32], 'N=%d ', ['origin', 'new', 'latency'], 'cholesky.res')

SS = os.getenv('SS')

run.run([12, 32], 'SBCONFIG=%s/ss-scheduler/configs/revel-1x2.sbmodel N=%s ' % (SS, '%d'), ['new'], 'cholesky.res')
run.run([12, 32], 'SBCONFIG=%s/ss-scheduler/configs/revel-2x2.sbmodel N=%s ' % (SS, '%d'), ['new'], 'cholesky.res')
run.run([12, 32], 'SBCONFIG=%s/ss-scheduler/configs/revel-3x2.sbmodel N=%s ' % (SS, '%d'), ['new'], 'cholesky.res')
run.run([12, 32], 'SBCONFIG=%s/ss-scheduler/configs/revel-3x3.sbmodel N=%s ' % (SS, '%d'), ['new'], 'cholesky.res')

#for i in [2, 4, 8]:
#    run.run([16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256], 'LANES=%d N=%s ' % (i, '%d'), ['latency'], 'cholesky.res')
#run.run([16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256], 'N=%s ' % ('%d'), ['new'], 'cholesky.res')
#run.run([64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256], 'N=%s ' % ('%d'), ['new'], 'cholesky.res')
