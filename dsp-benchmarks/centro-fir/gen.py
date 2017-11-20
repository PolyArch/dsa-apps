import sys, numpy, random
from math import sin, cos, pi 
n = int(sys.argv[1])
m = int(sys.argv[2])

if not (m % 2):
    exit()

numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)

a = numpy.random.rand(n).astype('complex64') + 1j * numpy.random.rand(n).astype('complex64')
b = numpy.random.rand(m / 2).astype('complex64') + 1j * numpy.random.rand(m / 2).astype('complex64')
b = numpy.concatenate((b, numpy.array([random.random() + 1j * random.random()]), b[::-1]))

numpy.savetxt('input.data', numpy.concatenate((a.flatten(), b.flatten())))
print 'input generated'

c = numpy.zeros((n - m + 1,)).astype('complex64')

for i in xrange(n - m + 1):
    for j in xrange(m):
        c[i] += a[i + j] * b[j]


numpy.savetxt('ref.data', c.flatten())
print 'output generated!'
