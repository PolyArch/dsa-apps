#!/usr/bin/env python3

import subprocess

cases = []
# N, M
cases.append(("datasets/very_small/dense_act.data", "datasets/very_small/wgt_ptr.data", "datasets/very_small/wgt_val.data", "datasets/very_small/wgt_index.data", 20, 15))
# cases.append(("datasets/fc6/pyfc6_dense_act_file.txt", "datasets/fc6/pyfc6_wgt_ptr.txt", "datasets/fc6/pyfc6_wgt_val.txt", "datasets/fc6/pyfc6_wgt_ind.txt", 9216, 4096))


for a, w1, w2, w3, N, M in cases:
    subprocess.call('make clean', shell=True)
    env1 = "dense_act_file=\\\\\\\"%s\\\\\\\" wgt_ptr_file=\\\\\\\"%s\\\\\\\" wgt_val_file=\\\\\\\"%s\\\\\\\" wgt_ind_file=\\\\\\\"%s\\\\\\\"" % (a, w1, w2, w3)
    env2 = " N=%d M=%d" % (N, M)
    env = env1 + env2
    # print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')

    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(N, M, "???")
    else:
        print(N, M, cycles)
