#!/usr/bin/env python3

import subprocess

cases = []
# index_file, shadow_file, circuit
# currently small one
# cases.append(("very_small"))
# cases.append(("water"))
# cases.append(("pigs"))
cases.append(("cpcs54.uai.ac"))
cases.append(("andes.uai.ac"))
# cases.append(("munin3.uai.ac"))
# cases.append(("mildew.uai.ac"))


subprocess.call('make clean', shell=True)

for f in cases:
    env = "dataset=\\\\\\\"%s\\\\\\\"" % (f)
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

    subprocess.call('make clean', shell=True)
    
# cases.append(("datasets/final_index.data", "datasets/final_shadow_index.data",
#    "datasets/final_circuit.data"))

