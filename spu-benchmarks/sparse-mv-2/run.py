import subprocess

cases = [[9216, 4096, 0.09, 0.351],
[4096, 4096, 0.09, 0.353],
[4096, 1000, 0.25, 0.375],
[25088, 4096, 0.04, 0.183],
[4096, 4096, 0.04, 0.375],
[4096, 600, 0.1, 1.0],
[600, 8191, 0.11, 1.0],
[1201, 2400, 0.11, 1.0]]

for case in cases:
    val = 0
    for i in range(10):
        raw = subprocess.check_output(['./mkl.exe'] + [str(i) for i in case]).decode('utf-8')
        raw = raw.lstrip('ticks: ').rstrip()
        val += int(raw)
    print(case, val / 10.)

