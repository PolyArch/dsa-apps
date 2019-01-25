#!/usr/bin/env python3

import subprocess

cases = []
# ACT file
cases.append(("75-17-5.uai.ac"))
# cases.append(("andes.uai.ac"))
# cases.append(("cpcs54.uai.ac"))
# cases.append(("tcc4e.uai.ac"))
# cases.append(("small_nnf.uai.ac"))
# cases.append(("water.uai.ac"))
# cases.append(("pigs.uai.ac"))
# cases.append(("munin3.uai.ac"))
# cases.append(("mildew.uai.ac"))

for f in cases:
    subprocess.call('make clean', shell=True)
    env = "dataset=\\\\\\\"%s\\\\\\\"" % (f)
    subprocess.call('make %s' % env, shell=True)
    subprocess.call('make run', shell=True)
    # raw = subprocess.check_output('make run', shell=True).decode('utf-8')
    # log_file = 'logs/%s.log' % (f)
    # f = open(log_file,'w+')
    # f.write('%s' %raw)
    # f.close()

    # print sparsity
