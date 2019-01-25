#!/usr/bin/env python3

import subprocess

cases = []
# file, N, M
# cases.append(("datasets/very_small.data", 84, 10))
# cases.append(("datasets/small_adult.data", 84, 100))
# cases.append(("datasets/diabetes.data", 8, 768))
cases.append(("datasets/diabetes.data", 120, 3840))

for f, n, m in cases:
    subprocess.call('make clean', shell=True)
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
        print(f, n, m, "???")
    else:
        print(f, n, m, cycles)

    # subprocess.call('make clean', shell=True)
