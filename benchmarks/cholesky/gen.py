# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, math, sys
n = int(sys.argv[1])
a = numpy.random.rand(n, n)
a = (numpy.dot(a, a.transpose()))
a = a + numpy.identity(n)
numpy.savetxt('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

L = numpy.identity(n)

origin = a.copy()
#print origin

for i in xrange(n):
    l = numpy.identity(n)
    div = math.sqrt(a[i, i])
    l[i, i] = div
    b = a[i, i + 1:]
    l[i + 1:, i] = b / l[i, i]

    aa = a.copy()
    aa[i, i] = 1
    aa[i, i + 1:] = numpy.zeros(n - i - 1)
    aa[i + 1:, i] = numpy.zeros(n - i - 1)
    aa[i + 1:, i + 1:] = a[i + 1:, i + 1:] - numpy.outer(b, b) / a[i, i]
    # Mathematically, it is L = L * l. However, in this case, we can just copy the corresponding
    # column to L, because the speciality of L and l
    L[i:, i] = l[i:, i]
    a = aa

numpy.testing.assert_allclose(origin, numpy.dot(L, L.transpose()))
print "Correctness check pass!"
numpy.savetxt('ref.data', L.flatten());
print "New data generated!"
