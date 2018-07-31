#!/usr/bin/env python3

import subprocess

cases = []
#cases.append((9216, 4096, 0.09, 0.351))
#cases.append((4096, 4096, 0.09, 0.353))
#cases.append((4096, 1000, 0.25, 0.375))
cases.append((25088, 4096, 0.04, 0.183))
cases.append((4096, 4096, 0.04, 0.375))
cases.append((4096, 600, 0.1, 1.0))
cases.append((600, 8191, 0.11, 1.0))
cases.append((1201, 2400, 0.11, 1.0))

simulate = []
simulate.append("SBCONFIG=$SS_TOOLS/configs/spu_merge_test.sbmodel")
simulate.append("FU_FIFO_LEN=15")
simulate.append("BACKCGRA=1")
simulate.append("gem5.opt")
simulate.append("/home/weng/ss-stack/gem5/configs/example/se.py")
simulate.append("--cpu-type=MinorCPU")
simulate.append("--l1d_size=2048kB")
simulate.append("--l1d_assoc=8")
simulate.append("--l1i_size=16kB --l2_size=16384kB --caches --cmd=./softbrain.exe")

simulate = ' '.join(simulate)

subprocess.check_output(['make', 'clean'])

for n, m, s0, s1 in cases:
    nn = (n - 1) / 64 + 1
    if nn % 8:
        nn -= nn % 8
    env = "N=%d M=%d s0=%f s1=%f " % (nn, m, s0, s1)
    subprocess.check_output('%s make softbrain.exe' % env, shell=True)
    raw = subprocess.check_output(simulate, shell=True).decode('utf-8')

    cycles = None
    for line in raw.split('\n'):
        if line.startswith("Cycles: "):
            cycles = int(line[8:].strip())
            break

    if cycles is None:
        print(n, m, s0, s1, "???")
    else:
        print(n, m, s0, s1, cycles / 1250)

    subprocess.check_output(['make', 'clean'])

#import subprocess
#
#cases = [[9216, 4096, 0.09, 0.351],
#[4096, 4096, 0.09, 0.353],
#[4096, 1000, 0.25, 0.375],
#[25088, 4096, 0.04, 0.183],
#[4096, 4096, 0.04, 0.375],
#[4096, 600, 0.1, 1.0],
#[600, 8191, 0.11, 1.0],
#[1201, 2400, 0.11, 1.0]]
#
#for case in cases:
#    val = 0
#    for i in range(10):
#        raw = subprocess.check_output(['./mkl.exe'] + [str(i) for i in case]).decode('utf-8')
#        raw = raw.lstrip('ticks: ').rstrip()
#        val += int(raw)
#    print(case, val / 10.)

