#!/usr/bin/env python3

import subprocess

cases = []
# file, N, M
cases.append(("datasets/small_cifar.train", 50, 8))
# cases.append(("datasets/very_small.data", 10, 136))
# cases.append(("datasets/small_mslr.train", 1000, 8))
# cases.append(("datasets/small_mslr.train", 1000, 8))
# cases.append(("datasets/small_connect.train", 1000, 8))
# cases.append(("datasets/small_higgs.train", 2000, 8))
# cases.append(("datasets/small_ltrc.train", 1000, 8))
# cases.append(("datasets/binned_small_mslr.train", 100, 136))
# cases.append(("datasets/mslr.train", 1000, 136))
# cases.append(("datasets/mslr.train", 250, 136))
# cases.append(("datasets/test", 250, 136))

subprocess.call('source ~/ss-stack/setup.sh', shell=True)

for f, n, m in cases:
    subprocess.call('make clean', shell=True)
    env = "file=\\\\\\\"%s\\\\\\\" N=%d M=%d" % (f, n, m)
    print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    
    log_file = 'logs/%s.log' % (n)
    f = open(log_file,'w+')
    f.write('%s' %raw)
    f.close()


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
