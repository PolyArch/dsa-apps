import numpy, cmath, sys, imp
output = imp.load_source('output', '../common/output.py')

n = int(sys.argv[1])
m = int(sys.argv[2])
a = numpy.random.rand(n) + 1j * numpy.random.rand(n)
b = numpy.random.rand(m) + 1j * numpy.random.rand(m)

output.print_complex_array('input.data', numpy.concatenate((a.flatten(), b.flatten())))
print("%d, %d Input generated!" % (n, m))

c = numpy.zeros(n - m + 1).astype('complex128')

for i in xrange(n - m + 1):
    c[i] = 0
    for j in xrange(m):
        c[i] += a[i + j] * b[j]

output.print_complex_array('ref.data', c.flatten());

