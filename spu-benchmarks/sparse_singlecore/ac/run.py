#!/usr/bin/env python3

import subprocess

cases = []
# index_file, shadow_file, circuit
# cases.append(("andes.uai.ac"))
# cases.append(("cpcs54.uai.ac"))
# cases.append(("tcc4e.uai.ac"))
# currently small one
# cases.append(("very_small"))
cases.append(("75-17-5.uai.ac"))
# cases.append(("water"))
# cases.append(("pigs"))
# cases.append(("munin3.uai.ac"))
# cases.append(("mildew.uai.ac"))

subprocess.call('make clean', shell=True)

for f in cases:
    env = "dataset=\\\\\\\"%s\\\\\\\"" % (f)
    print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')

    print(raw)
    
    log_file = 'logs/%s.log' % (f)
    f = open(log_file,'w+')
    f.write('%s' %raw)
    f.close()


    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(f, "???")
    else:
        print(f, cycles)

    subprocess.call('make clean', shell=True)
    
# cases.append(("datasets/final_index.data", "datasets/final_shadow_index.data",
#    "datasets/final_circuit.data"))

