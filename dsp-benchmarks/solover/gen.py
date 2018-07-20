import numpy as np

output = imp.load_source('output', '../common/output.py')

n = int(sys.argv[1])
#numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)

a = numpy.random.rand(n, n).astype('complex64') + 1j * numpy.random.rand(n, n).astype('complex64')
for i in range(n):
    for j in range(i + 1, n):
        a[i, j] = 0. + 0.j
b = numpy.random.rand(n).astype('complex64') + 1j * numpy.random.rand(n).astype('complex64')
output.print_complex_array('input.data', numpy.concatenate((a.flatten(), b.flatten())))
print('input generated')

c = numpy.dot(a, b)
output.print_complex_array('ref.data', c.flatten())
print('output generated!')

