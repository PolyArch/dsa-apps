#!/usr/bin/env python3

import subprocess

cases = []
# index_file, shadow_file, circuit
# currently small one
cases.append(("datasets/final_index.data", "datasets/final_shadow_index.data",
    "datasets/final_circuit.data"))

subprocess.call('make clean', shell=True)

for f, n, m in cases:
    env = "index_file=\\\\\\\"%s\\\\\\\" shadow_file=\\\\\\\"%s\\\\\\\" circuit_file=\\\\\\\"%s\\\\\\\"" % (f, n, m)
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
