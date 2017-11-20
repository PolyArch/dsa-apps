import sys, numpy
from math import sin, cos, pi 
m = int(sys.argv[1])
n = int(sys.argv[2])
numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)

a = numpy.random.rand(m, n).astype('complex64') + 1j * numpy.random.rand(m, n).astype('complex64')
b = numpy.random.rand(n, 1).astype('complex64') + 1j * numpy.random.rand(n, 1).astype('complex64')

numpy.savetxt('input.data', numpy.concatenate((a.flatten(), b.flatten())))
print 'input generated'

c = numpy.dot(a, b)
numpy.savetxt('ref.data', c)
print 'output generated!'
