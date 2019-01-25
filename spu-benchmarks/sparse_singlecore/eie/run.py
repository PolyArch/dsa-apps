#!/usr/bin/env python3

import subprocess

cases = []
# layer, N, M
# cases.append(("very_small", 20, 15))
# cases.append(("vggfc12", 4096, 25088))
# cases.append(("vggfc13", 4096, 4096))
# cases.append(("alexfc6", 4096, 9216))
cases.append(("resnetfc1", 1000, 512))
# cases.append(("interfc13", 256, 4096))
# cases.append(("interfc13", 32, 4096))


for l, N, M in cases:
    subprocess.call('make clean', shell=True)
    # env1 = " N=%d M=%d" % (N, M)
    env1 = " N=%d M=%d" % (N, 4)
    env2 = " layer_name=\\\\\\\"%s\\\\\\\"" % (l)
    env = env1 + env2
    # print(env)
    subprocess.call('make %s' % env, shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    log_file = 'logs/%s.log' % (l)
    f = open(log_file,'w+')
    f.write('%s' %raw)
    f.close()


    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(N, M, "???")
    else:
        print(N, M, cycles)
