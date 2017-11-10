import imp
output = imp.load_source('output', '../common/output.py')
import numpy, cmath, sys
from scipy import linalg

numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)
n = int(sys.argv[1])
f = file('debug.data', 'w')
_a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
_a = _a.astype('complex64')
output.print_complex_array('input.data', _a.flatten())
print 'Data generated!'
#print numpy.linalg.svd(_a)

t = numpy.dot(numpy.conj(_a.transpose()), _a)
h = t.copy()
f.write('asta:\n' + str(t) + '\n')

left = numpy.identity(n).astype('complex128')
right = numpy.identity(n).astype('complex128')
for i in xrange(n - 1):
    x = h[i + 1:, i].copy()
    v = x.copy()
    v[0] += cmath.exp(1j * cmath.phase(v[0])) * numpy.linalg.norm(v)
    v = v / numpy.linalg.norm(v)
    w = numpy.dot(numpy.conj(x), v) / numpy.dot(numpy.conj(v), x)
    H = numpy.identity(n - i - 1) - (1 + w) * numpy.outer(v, numpy.conj(v))
    h[i + 1:, i:] = numpy.dot(H, h[i + 1:, i:])
    #f.write('ph:\n' + str(h) + '\n')
    h[:, i + 1:] = numpy.dot(h[:, i + 1:], H)
    #f.write('hp:\n' + str(h) + '\n')
    left[i+1:,:] = numpy.dot(H, left[i+1:,:])
    right[:,i+1:] = numpy.dot(right[:,i+1:], H)

f.write('hessenberg:\n' + str(h) + '\n')
f.write('transform:\n' + str(right) + '\n')

#numpy.testing.assert_allclose(numpy.dot(numpy.dot(right, h), left), t, atol = 1e-5)

_h = h.copy()
V = numpy.identity(n)
cycles = 0
for i in xrange(1000):
    cycles += 1
    #numpy.testing.assert_allclose(numpy.conj(R.transpose()), R, atol = 1e-5);
    Q = numpy.identity(n).astype('complex128')
    d = (h[n - 2, n - 2] - h[n - 1, n - 1]) / 2.
    R = h
    for i in xrange(n - 1):
        x = h[i:i+2, i].copy()
        v = x.copy()
        v[0] += cmath.exp(1j * cmath.phase(v[0])) * numpy.linalg.norm(v)
        norm = numpy.linalg.norm(v)
        v = v / norm
        w = numpy.dot(numpy.conj(x), v) / numpy.dot(numpy.conj(v), x)
        H = numpy.identity(2) - (1 + w) * numpy.outer(v, numpy.conj(v))
        h[i:i+2,i:min(i+3,n)] = numpy.dot(H, h[i:i+2,i:min(i+3,n)])
        #f.write('R:\n' + str(h) + '\n')
        Q[:min(i+3,n),i:i+2] = numpy.dot(Q[:min(i+3,n),i:i+2], H)
        #f.write('Q:\n' + str(Q) + '\n')
    f.write('Q:\n' + str(Q) + '\n')
    f.write('R:\n' + str(R) + '\n')
    h = numpy.dot(R, Q)# + mu * numpy.identity(n)
    V = numpy.dot(V, Q)
    f.write('RQ:\n' + str(h) + '\n')
    #f.write('V:\n' + str(V) + '\n')
    #print sum(h.flatten() - prev.flatten())
    if (sum(abs(i) > 2e-3 for i in h.flatten())) <= n:
        break

print cycles

#numpy.testing.assert_allclose(numpy.dot(_h, V),  numpy.dot(V, h), atol = 1e-5)
V = numpy.dot(right, V)
#numpy.testing.assert_allclose(numpy.dot(t, V), numpy.dot(V, h), atol = 1e-5)
S = numpy.array([cmath.sqrt(i).real for i in numpy.diag(h)])

#f.write('V:\n' + str(V) + '\n')

sigma = numpy.zeros((n, n))
for i in xrange(n):
    sigma[i, i] = cmath.sqrt(h[i, i]).real

U = numpy.dot(_a, V)
#f.write("U':\n" + str(U) + '\n')
for i in xrange(n):
    U[:,i] /= S[i]

if U[0, 0].real < 0:
    U = -1 * U
    V = -1 * V

#numpy.testing.assert_allclose(numpy.dot(U, numpy.conj(U.transpose())), numpy.identity(n), atol = 1e-5)
#numpy.testing.assert_allclose(numpy.dot(V, numpy.conj(V.transpose())), numpy.identity(n), atol = 1e-5)
#numpy.testing.assert_allclose(numpy.dot(U, numpy.dot(sigma, numpy.conj(V.transpose()))), _a, atol = 1e-5)
print 'Correctness check pass!'

#f.write('U:\n' + str(U) + '\n')
#f.write('S:\n' + str(S) + '\n')
#f.write('V*:\n' + str(numpy.conj((V).transpose())) + '\n')
#print U
#print S
#print numpy.conj(V.transpose())
output.print_complex_array('ref.data', numpy.concatenate((U.flatten(), S.flatten(), numpy.conj(V.transpose()).flatten())))
#numpy.savetxt('ref.data', numpy.concatenate((U.flatten(), S.flatten(), numpy.conj(V.transpose()).flatten())))
print 'Ref data generated!'

