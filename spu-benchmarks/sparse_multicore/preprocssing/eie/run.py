#!/usr/bin/env python3

import subprocess

cases = []
# wgt_file, act_file
# layer, N, M
# cases.append(("very_small", 20, 15))
# cases.append(("vggfc12", 4096, 25088))
cases.append(("vggfc13", 4096, 4096))
# cases.append(("resnetfc1", 1000, 512))

for n, N, M in cases:
    subprocess.call('make clean', shell=True)
    env2 = "net_name=\\\\\\\"%s\\\\\\\"" % (n)
    env3 = " N=%d M=%d" % (N, M)
    env = env2 + env3
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('make run', shell=True)
    # raw = subprocess.check_output('make run', shell=True).decode('utf-8')

    # print sparsity
    # cycles = None
    # for line in raw.split('\n'):
    #     if line.startswith("Cycles: "):
    #         cycles = int(line[8:].strip())
    #         break

    # if cycles is None:
    #     print(Ni, Nx, Tx, Nn, Tn, Kx, "???")
    # else:
    #     print(Ni, Nx, Tx, Nn, Tn, Kx, cycles)

# cases.append(("datasets/wgt_val.data", "datasets/wgt_index.data", "datasets/wgt_ptr.data", "datasets/act_val.data", "datasets/act_index.data", "datasets/act_ptr.data", 96, 55, 55, 7, 7, 96, 356, 64, 5, 5))
