import sys, numpy, imp
from math import sin, cos, pi 
output = imp.load_source('output', '../common/output.py')

n = int(sys.argv[1])

numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)

a = numpy.random.rand(n)
for i in xrange(0, n, 2):
    a[i] = 1.0
output.print_complex_array('input.data', a)
print 'input generated'

b = numpy.sqrt(a)
output.print_complex_array('ref.data', b.flatten())
print 'output generated!'
