#!/usr/bin/env python3

import subprocess

cases = []
# wgt_val_file, wgt_ind_file, wgt_ptr_file, act_val_file, act_ind_file,
# act_ptr_file, Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky
# will work just for single core
# cases.append(("very_small", 1, 10, 10, 10, 10, 1, 1, 3, 3))
# cases.append(("resnet-conv1", 64, 56, 56, 7, 7, 128, 32, 3, 3)) # Tn=20?
# cases.append(("resnet-conv2", 64, 56, 56, 7, 7, 128, 32, 3, 3)) # Tn=20?
# cases.append(("alexconv1", 3, 224, 224, 28, 28, 64, 1, 11, 11)) # Tn=20?
# cases.append(("vggconv3", 64, 112, 112, 14, 14, 128, 1, 3, 3)) # TODO: decide Tn
cases.append(("vggconv4", 128, 112, 112, 14, 14, 128, 1, 3, 3)) # TODO: decide Tn
# cases.append(("alexconv2", 64, 27, 27, 3, 3, 192, 1, 5, 5)) # Tn=20?

# cases.append(("datasets/wgt_val.data", "datasets/wgt_index.data", "datasets/wgt_ptr.data", "datasets/act_val.data", "datasets/act_index.data", "datasets/act_ptr.data", 96, 55, 55, 7, 7, 96, 356, 64, 5, 5))



# for w1, w2, w3, a1, a2, a3, Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky in cases:
for n, Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky in cases:
    subprocess.call('make clean', shell=True)
    # env1 = "wgt_val_file=\\\\\\\"%s\\\\\\\" wgt_ind_file=\\\\\\\"%s\\\\\\\" wgt_ptr_file=\\\\\\\"%s\\\\\\\"" % (w1, w2, w3)
    # env2 = " act_val_file=\\\\\\\"%s\\\\\\\" act_ind_file=\\\\\\\"%s\\\\\\\" act_ptr_file=\\\\\\\"%s\\\\\\\"" % (a1, a2, a3)
    env1 = "net_name=\\\\\\\"%s\\\\\\\"" % (n)
    env3 = " Ni=%d Nx=%d Ny=%d Tx=%d Ty=%d Nn=%d Tn=%d Kx=%d Ky=%d" % (Ni, Nx, Ny, Tx, Ty, Nn, Tn, Kx, Ky)
    env = env1 + env3
    # env = env1 + env2 + env3
    print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    print(raw)
    
    log_file = 'logs/%s.log' % (n)
    f = open(log_file,'w+')
    f.write('%s' %raw)
    f.close()


    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Ticks: "):
            cycles = int(line[7:].strip())
            break

    if cycles is None:
        print(Ni, Nx, Tx, Nn, Tn, Kx, "???")
    else:
        print(Ni, Nx, Tx, Nn, Tn, Kx, cycles)
