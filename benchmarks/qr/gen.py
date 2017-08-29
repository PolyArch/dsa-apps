# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, math, sys
n = int(sys.argv[1])
a = numpy.random.rand(n, n)
a = a + numpy.identity(n)
numpy.savetxt('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

Q = numpy.identity(n)
R = a.copy()

#print origin

for i in xrange(n):
    w = numpy.concatenate((numpy.zeros(i), R[i:, i].copy()))
    w[i] += (1 if w[i] >= 0 else -1) * numpy.linalg.norm(w)
    w = w / numpy.linalg.norm(w)
    H = numpy.identity(n) - 2 * numpy.outer(w, w)
    R = numpy.dot(H, R)
    Q = numpy.dot(Q, H)

numpy.testing.assert_allclose(a, numpy.dot(Q, R), rtol = 1e-5)
numpy.testing.assert_allclose(numpy.identity(n), numpy.dot(Q, Q.transpose()), atol = 1e-5)

print "Correctness check pass!"
numpy.savetxt('ref.data', numpy.concatenate((Q.flatten(), R.flatten())));
print "New data generated!"
