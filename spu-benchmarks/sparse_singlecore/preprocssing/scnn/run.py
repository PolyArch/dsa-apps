#!/usr/bin/env python3

import subprocess

cases = []
# wgt_file, act_file
# can I just have prefix (net name -- other than that everything same)
# Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky
# will work just for single core
# cases.append(("very_small", 1, 10, 10, 10, 10, 1, 1, 3, 3))
# cases.append(("resnet-conv1", 64, 56, 56, 7, 7, 128, 32, 3, 3)) # Tn=20?
# cases.append(("resnet-conv2", 64, 56, 56, 7, 7, 128, 16, 3, 3)) # Tn=20?
# cases.append(("alexconv1", 3, 224, 224, 28, 28, 64, 1, 11, 11)) # Tn=20?
# cases.append(("alexconv2", 64, 27, 27, 3, 3, 192, 96, 5, 5)) # Tn=20?
# cases.append(("vggconv3", 64, 112, 112, 14, 14, 128, 4, 3, 3)) # TODO: decide Tn
cases.append(("vggconv4", 128, 112, 112, 14, 14, 128, 4, 3, 3)) # TODO: decide Tn


for n, Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky in cases:
    subprocess.call('make clean', shell=True)
    env1 = "net_name=\\\\\\\"%s\\\\\\\"" % (n)
    env2 = " Ni=%d Nx=%d Ny=%d Tx=%d Ty=%d Nn=%d Tn=%d Kx=%d Ky=%d" % (Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky)
    env = env1 + env2
    print(env)
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('make run', shell=True)
    # raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    
    # log_file = 'logs/%s.log' % (n)
    # f = open(log_file,'w+')
    # f.write('%s' % (raw))
    # f.close()


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

    # subprocess.call('make clean', shell=True)

# cases.append(("input_datasets/very_small/wgt.data",
#    "input_datasets/very_small/act.data", 1, 10, 10, 10, 10, 1, 1, 3, 3))

