#!/usr/bin/env python3

import subprocess

cases = []
# data_file, csr_file, V, E
cases.append(("datasets/rome99_csr","datasets/rome_99",3352,8859))
# cases.append(("datasets/fb_csr","datasets/fb_99",50516,1638612))



for data_file, csr_file, v, e  in cases:
    subprocess.call('make clean', shell=True)
    subprocess.call('make cleandata', shell=True)
    env1 = "V=%d E=%d" % (v,e)
    env2 = " data_file=\\\\\\\"%s\\\\\\\" csr_file=\\\\\\\"%s\\\\\\\"" % (data_file, csr_file)
    env = env1 + env2
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('make csr %s' % env, shell=True)
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
