#!/usr/bin/env python3

import subprocess

cases = []
# file, N, M
# cases.append(("datasets/small_connect.train", 126, 1000))
# cases.append(("datasets/small_higgs.train", 28, 2000))
# cases.append(("datasets/small_cifar.train", 3072, 50))
# cases.append(("datasets/small_ltrc.train", 700, 1000))
# cases.append(("datasets/small_mslr.train", 136, 1000))
cases.append(("datasets/very_small.data", 84, 10))
# cases.append(("datasets/small_adult.data", 84, 100))
# cases.append(("datasets/diabetes.data", 8, 768))
# cases.append(("datasets/connect-4.data", 126, 67584))

for f, n, m in cases:
    subprocess.call('make clean', shell=True)
    env = "file=\\\\\\\"%s\\\\\\\" N=%d M=%d" % (f, n, m)
    print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    
    log_file = 'logs/%s.log' % (f)
    f1 = open(log_file,'w+')
    f1.write('%s' %raw)
    f1.close()


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
