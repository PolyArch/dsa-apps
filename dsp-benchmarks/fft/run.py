#!/usr/bin/env python3.5
import imp

run = imp.load_source('run', '../common/run.py')

run.run([64, 128, 256, 512, 1024, 2048], 'N=%d ', ['origin', 'new'])
