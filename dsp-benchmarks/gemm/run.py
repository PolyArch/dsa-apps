#!/usr/bin/env python3.5
import imp

run = imp.load_source('run', '../common/run.py')

run.run([12, 24, 48, 120], 'N=%d M=16 P=64 ', ['origin', 'new'])
