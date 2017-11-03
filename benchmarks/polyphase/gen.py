import numpy, cmath, sys
n = int(sys.argv[1])
m = int(sys.argv[2])
d = int(sys.argv[3])
#a = numpy.random.rand(n) + 1j * numpy.random.rand(n)
#b = numpy.random.rand(m) + 1j * numpy.random.rand(m)
a = numpy.array(range(1, n + 1)) + 1j * numpy.zeros(n)
b = numpy.array(range(1, m + 1)) + 1j * numpy.zeros(m)

numpy.savetxt('input.data', numpy.concatenate((a.flatten(), b.flatten())))
print("%d, %d Input generated!" % (n, m))

c = numpy.zeros(n - m + 1).astype('complex128')

for i in xrange(n - m + 1):
    c[i] = 0
    for j in xrange(m):
        c[i] += a[i + j] * b[j]

numpy.savetxt('ref.data', c[::d].flatten());

