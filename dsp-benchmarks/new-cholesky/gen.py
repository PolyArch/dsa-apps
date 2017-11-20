import imp
output = imp.load_source('output', '../common/output.py')

def round(a):
    return a.real.astype('int16') + a.imag.astype('int16') * 1j


# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, cmath, sys
scale = 50
n = int(sys.argv[1])
a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
a = (numpy.dot(a, numpy.conj(a.transpose())))
#a = a + numpy.identity(n)
#a *= scale * scale;
#a = round(a)
output.print_complex_array('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

L = numpy.identity(n).astype('complex')

origin = a.copy()
#print origin

for i in xrange(n):
    l = numpy.identity(n).astype('complex')
    div = cmath.sqrt(a[i, i])
    l[i, i] = div
    b = a[i, i + 1:]
    l[i + 1:, i] = b / l[i, i]

    aa = a.copy()
    aa[i, i] = 1
    aa[i, i + 1:] = numpy.zeros(n - i - 1)
    aa[i + 1:, i] = numpy.zeros(n - i - 1)
    aa[i + 1:, i + 1:] = a[i + 1:, i + 1:] - numpy.outer(numpy.conj(b), b) / a[i, i]
    # Mathematically, it is L = L * l. However, in this case, we can just copy the corresponding
    # column to L, because the speciality of L and l
    L[i:, i] = l[i:, i]
    #L = numpy.dot(L, l)
    a = aa

numpy.testing.assert_allclose(origin, numpy.dot(numpy.conj(L), L.transpose()), rtol = 1e-4)
print "Correctness check pass!"
output.print_complex_array('ref.data', L.flatten());
print "New data generated!"

