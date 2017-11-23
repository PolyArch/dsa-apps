import imp
output = imp.load_source('output', '../common/output.py')

# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, cmath, sys
n = int(sys.argv[1])
a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
a = a + numpy.identity(n)

output.print_complex_array('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

Q = numpy.identity(n)
R = a.copy()

#print origin

for i in xrange(n):
    x = numpy.concatenate((numpy.zeros(i), R[i:, i].copy()))
    v = x.copy()
    #print v
    v[i] += cmath.exp(1j * cmath.phase(v[i])) * numpy.linalg.norm(v)
    v = v / numpy.linalg.norm(v)
    w = numpy.dot(numpy.conj(x), v) / numpy.dot(numpy.conj(v), x)
    H = numpy.identity(n) - (1 + w) * numpy.outer(v, numpy.conj(v))
    R = numpy.dot(H, R)
    #print R[i:,i:]
    Q = numpy.dot(Q, H)

numpy.testing.assert_allclose(a, numpy.dot(Q, R), rtol = 1e-5)
numpy.testing.assert_allclose(numpy.identity(n), numpy.dot(Q, numpy.conj(Q.transpose())), atol = 1e-5)

print "Correctness check pass!"
output.print_complex_array('ref.data', numpy.concatenate((Q.flatten(), R.flatten())));
print "New data generated!"
