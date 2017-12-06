#!/usr/bin/env python3
import imp

run = imp.load_source('run', '../common/run.py')

run.run([37, 73, 147, 199], 'M=%d ', ['origin', 'new', 'spad'])
