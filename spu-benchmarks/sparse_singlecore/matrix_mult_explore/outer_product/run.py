#!/usr/bin/env python3

import subprocess

cases = []
# wgt_sp, n_sp
# cases.append((0.07,0.3))
# cases.append((0.08,0.5))
# cases.append((0.09,0.7))
# cases.append((0.09,0.9))
# cases.append((0.1,1))
# cases.append((0.67,0.3))
# cases.append((0.80,0.5))
# cases.append((0.86,0.7))
# cases.append((0.75,0.75))
cases.append((0.25,0.25))
# cases.append((0.50,0.50))
# cases.append((1,1))



for syn_sp, act_sp  in cases:
    subprocess.call('make clean', shell=True)
    subprocess.call('make cleandata', shell=True)
    env = " act_sp=%0.2f syn_sp=%0.2f" % (act_sp,syn_sp)
    # print(env)
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('./a.out', shell=True)
    raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    
    log_file = 'logs/%s.log' % ('x_%0.2f' % (syn_sp))
    f = open(log_file,'w+')
    f.write('%s' %raw)
    f.close()


    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(syn_sp, act_sp, "???")
    else:
        print(syn_sp, act_sp, cycles)
