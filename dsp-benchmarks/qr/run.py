#!/usr/bin/env python3.5
import imp

run = imp.load_source('run', '../common/run.py')

run.run([8, 12, 16, 24, 32], 'N=%d ', ['origin', 'new'])
