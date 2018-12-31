#!/usr/bin/env python3

import subprocess

cases = []
# file, N, M
cases.append(("small_cifar.train", 50, 3073, 8))
# cases.append(("datasets/very_small.data", 10, 136))
cases.append(("small_connect.train", 1000, 136, 8))
cases.append(("small_mslr.train", 1000, 136, 8))
cases.append(("small_ltrc.train", 1000, 700, 8))
cases.append(("small_higgs.train", 2000, 28, 8))
# cases.append(("datasets/mslr.train", 250, 136))
# cases.append(("datasets/test", 250, 136))

subprocess.call('source ~/ss-stack/setup.sh', shell=True)

for f, n, m, t in cases:
    subprocess.call('make clean', shell=True)
    env = "dataset=\\\\\\\"%s\\\\\\\" N=%d M=%d feat_needed=%d" % (f, n, m, t)
    print(env)
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('make run', shell=True)
