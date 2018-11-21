#!/usr/bin/env python3

import subprocess

cases = []
# wgt_val_file, wgt_ind_file, wgt_ptr_file, act_val_file, act_ind_file,
# act_ptr_file, Ni, Nx, Ny, Tx, Ty, Ni, Nn, Tn, Kx, Ky
# will work just for single core
cases.append(("datasets/very_small/wgt_val.data",
    "datasets/very_small/wgt_index.data", "datasets/very_small/wgt_ptr.data",
    "datasets/very_small/act_val.data", "datasets/very_small/act_index.data",
    "datasets/very_small/act_ptr.data", 1, 10, 10, 10, 10, 1, 1, 1, 3, 3))
# cases.append(("datasets/wgt_val.data", "datasets/wgt_index.data", "datasets/wgt_ptr.data", "datasets/act_val.data", "datasets/act_index.data", "datasets/act_ptr.data", 96, 55, 55, 7, 7, 96, 356, 64, 5, 5))


subprocess.call('make clean', shell=True)

for w1, w2, w3, a1, a2, a3, Ni, Nx, Ny, Tx, Ty, Ni, Nn, Tn, Kx, Ky in cases:
    env1 = "wgt_val_file=\\\\\\\"%s\\\\\\\" wgt_ind_file=\\\\\\\"%s\\\\\\\" wgt_ptr_file=\\\\\\\"%s\\\\\\\"" % (w1, w2, w3)
    env2 = " act_val_file=\\\\\\\"%s\\\\\\\" act_ind_file=\\\\\\\"%s\\\\\\\" act_ptr_file=\\\\\\\"%s\\\\\\\"" % (a1, a2, a3)
    env3 = " Ni=%d Nx=%d Ny=%d Tx=%d Ty=%d Ni=%d Nn=%d Tn=%d Kx=%d Ky=%d" % (Ni, Nx, Ny, Tx, Ty, Ni, Nn, Tn, Kx, Ky)
    env = env1 + env2 + env3
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
