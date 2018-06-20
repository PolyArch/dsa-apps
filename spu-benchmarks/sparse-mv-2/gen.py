import numpy, sys

try:
    n, m, s0, s1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
except:
    n, m, s0, s1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), 1.0

a = numpy.random.randn(n, m)

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
    if len(line) % 2:
        line.append('1 0.0')
    line.append(('-1 -1'))
    return line

for i in range(n):
    print ' '.join(compress_line(a[i], m, s0))

print

v = numpy.random.randn(m)
print ' '.join(compress_line(v, m, s1))

