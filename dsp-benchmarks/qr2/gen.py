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

tau = numpy.zeros((n, ), dtype = 'complex128')

#print origin
q = numpy.identity(n, dtype = 'complex128')

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

    h = numpy.identity(n - i, dtype = 'complex128') - tau[i] * numpy.outer(w, numpy.conj(w))
    #print h
    #print numpy.dot(h, numpy.conj(h).transpose())
    #print numpy.dot(h, a[i:, i:])
    #print

    a[i:,i+1:] -= tau[i] * numpy.outer(w, numpy.dot(numpy.conj(w), a[i:,i+1:]))
    q[:,i:] = numpy.dot(q[:,i:], h)

print numpy.dot(numpy.conj(q).transpose(), origin)

print a
print tau

output.print_complex_array('ref.data', numpy.concatenate((a.flatten(), tau)))
print "New data generated!"

