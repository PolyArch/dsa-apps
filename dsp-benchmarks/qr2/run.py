#!/usr/bin/env python3
import imp

run = imp.load_source('run', '../tools/run.py')

run.run([12, 16, 24, 32], 'N=%d ', ['new'], True)