import imp
output = imp.load_source('output', '../common/output.py')

# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, cmath, sys
n = int(sys.argv[1])
a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
a = a + numpy.identity(n)

origin = a.copy()

output.print_complex_array('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

tau = numpy.zeros((n - 1, ), dtype = 'complex128')

for i in xrange(n - 1):
    w = a[i:,i].copy()
    normx = numpy.linalg.norm(w)
    s = -w[0] / cmath.sqrt(w[0].conjugate() * w[0])
    u1 = w[0] - s * normx
    w /= u1
    w[0] = 1 + 0j
    a[i, i] = s * normx
    a[i+1:,i] = w[1:]
    tau[i] = -s.conjugate() * u1 / normx

    v = tau[i] * numpy.dot(numpy.conj(w), a[i:,i+1:])
    a[i,i+1:]    -= v
    a[i+1:,i+1:] -= numpy.outer(w[1:], v)

q = numpy.identity(n, dtype = 'complex128')
for i in xrange(n - 2, -1, -1):
    w = a[i+1:,i]
    v = numpy.dot(numpy.conj(w), q[i+1:,i+1:])
    q[i,i] = 1 - tau[i]
    q[i,i+1:] = -tau[i] * v
    q[i+1:,i+1:] -= tau[i] * numpy.outer(w, v)
    q[i+1:,i] = a[i+1:,i] * -tau[i]

#print a
#print numpy.dot(numpy.conj(q.transpose()), origin)

output.print_complex_array('ref.data', numpy.concatenate((a.flatten(), tau, q.flatten())))

print "New data generated!"

