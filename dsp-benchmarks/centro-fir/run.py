#!/usr/bin/env python3
import imp

run = imp.load_source('run', '../tools/run.py')

run.run([37, 73, 147, 199], 'M=%d ', ['origin', 'new', 'latency'])
run.run([(164, 37), (200, 73), (274, 147), (326, 199)], 'N=%d M=%d ', ['origin'], True, False)
