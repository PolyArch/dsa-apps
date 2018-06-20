import numpy, sys

try:
    n, m, s0, s1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
except:
    n, m, s0, s1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), 1.0

def compress_line(a, n, ratio):
    last = 0
    line = []
    for j in range(m):
        if numpy.random.random() < ratio:
            while (j - last) >= 2 ** 4:
                line.append('%d %f' % (2 ** 4 - 1, 0.0))
                last += 2 ** 4 - 1
            line.append('%d %f' % (j - last, a[j]))
            last = j
        else:
            a[j] = 0.0
    if len(line) % 2:
        line.append('1 0.0')
    line.append(('-1 -1'))
    return line

a = numpy.random.randn(n, m)
v = numpy.random.randn(m)

with open('input.data', 'w') as f_input:
    for i in range(n):
        f_input.write(' '.join(compress_line(a[i], m, s0)) + '\n')
    f_input.write(' '.join(compress_line(v, m, s1)))


with open('output.data', 'w') as f_output:
    ref = numpy.dot(a, v)
    f_output.writelines('\n'.join('%.3f' % i for i in ref))
