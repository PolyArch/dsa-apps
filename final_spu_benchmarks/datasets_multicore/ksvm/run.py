#!/usr/bin/env python3

import subprocess

cases = []
# file, N, M
cases.append(("datasets/very_small.data", 84, 10))
# cases.append(("datasets/small_adult.data", 84, 100))

subprocess.call('make clean', shell=True)

for f, n, m in cases:
    env = "file=\\\\\\\"%s\\\\\\\" N=%d M=%d" % (f, n, m)
    print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')

    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(n, m, s0, s1, "???")
    else:
        print(n, m, s0, s1, cycles)

    subprocess.call('make clean', shell=True)
