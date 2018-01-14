import numpy, cmath, sys, imp
from scipy import linalg

output = imp.load_source('output', '../common/output.py')

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

right = numpy.identity(n).astype('complex128')
for i in xrange(n - 1):
    x = h[i + 1:, i].copy()
    v = x.copy()
    v[0] += cmath.exp(1j * cmath.phase(v[0])) * numpy.linalg.norm(v)
    v = v / numpy.linalg.norm(v)
    #f.write('v%d:%s\n' % (i, str(v)))
    w = numpy.dot(numpy.conj(x), v) / numpy.dot(numpy.conj(v), x)
    assert abs(w.real - 1) < 1e-5
    # H = numpy.identity(n - i - 1) - (1 + w) * numpy.outer(v, numpy.conj(v)) DEPRECATED in fine grained implementation
    # h[i + 1:, i:] = numpy.dot(H, h[i + 1:, i:])          (1)
    # h[:, i + 1:] = numpy.dot(h[:, i + 1:], H)            (2)
    # right[1:, i + 1:] = numpy.dot(right[1:, i + 1:], H)  (3)
    # fine grained representitive

    # (1)
    temp = 2 * numpy.dot(numpy.conj(v), h[i + 1:, i:])
    h[i + 1:, i:] -= numpy.outer(v, temp)
    f.write('p0 %d:\n%s\n' % (i, str(temp)))
    f.write('p1 %d:\n%s\n' % (i, str(h[i + 1:, i:])))
    # (2)
    temp = 2 * numpy.dot(h[i:, i + 1:], v)
    h[i:, i + 1:] -= numpy.outer(temp, numpy.conj(v))
    f.write('p2 %d:\n%s\n' % (i, str(temp)))
    f.write('p3 %d:\n%s\n' % (i, str(h[i:, i + 1:])))
    # (3)
    temp = numpy.dot(right[1:, i + 1:], v) * 2
    right[1:, i + 1:] -= numpy.outer(temp, numpy.conj(v))
    f.write('p\'0 %d:\n%s\n' % (i, str(temp)))
    f.write('inv %d:\n%s\n' % (i, str(right[1:, i + 1:])))

f.write('hessenberg:\n' + str(h) + '\n')
f.write('transform:\n' + str(right) + '\n')

_h = h.copy()
V = numpy.identity(n)

for TOTAL in xrange(1000):
    Q = numpy.identity(n).astype('complex128')
    R = h
    for i in xrange(n - 1):
        x = h[i:i+2, i].copy()
        v = x.copy()
        v[0] += cmath.exp(1j * cmath.phase(v[0])) * numpy.linalg.norm(v)
        norm = numpy.linalg.norm(v)
        v = v / norm
        w = numpy.dot(numpy.conj(x), v) / numpy.dot(numpy.conj(v), x)
        assert abs(1 - w.real) < 1e-5
        H = numpy.identity(2) - (1 + w) * numpy.outer(v, numpy.conj(v))
        #H[0, 0] == H[1, 1]
        assert abs(H[0, 0] + H[1, 1]) < 1e-5
        f.write('v:\n%s\n' % str(v))
        f.write('H:\n%s\n' % str(H))
        #print h[i:i+2,i:min(i+3,n)]
        h[i:i+2,i:min(i+3,n)] = numpy.dot(H, h[i:i+2,i:min(i+3,n)])
        #print h[i:i+2,i:min(i+3,n)]
        #Q[:min(i+2,n),i:i+2] = numpy.dot(Q[:min(i+2,n),i:i+2], H)
        #fine grained rep
        Q[:min(i+2,n)-1, i + 1] = Q[:min(i+2,n) - 1,i] * H[0, 1]
        Q[:min(i+2,n)-1, i]    *= H[0, 0]
        Q[min(i+2,n) - 1,i]     = numpy.conj(H[0, 1])
        Q[min(i+2,n) - 1,i + 1] = -H[0, 0]
    #print 
    f.write('Q:\n' + str(Q) + '\n')
    f.write('R:\n' + str(R) + '\n')
    h = numpy.dot(R, Q)
    V = numpy.dot(V, Q)
    f.write('RQ:\n' + str(h) + '\n')
    #f.write('V:\n' + str(V) + '\n')
    if (sum(abs(i) > 1e-5 for i in h.flatten())) <= n:
        print 'Total',  TOTAL
        break

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

numpy.testing.assert_allclose(numpy.dot(U, numpy.conj(U.transpose())), numpy.identity(n), atol = 1e-3)
numpy.testing.assert_allclose(numpy.dot(V, numpy.conj(V.transpose())), numpy.identity(n), atol = 1e-3)
numpy.testing.assert_allclose(numpy.dot(U, numpy.dot(sigma, numpy.conj(V.transpose()))), _a, atol = 1e-3)
print 'Correctness check pass!'

#f.write('U:\n' + str(U) + '\n')
#f.write('S:\n' + str(S) + '\n')
#f.write('V*:\n' + str(numpy.conj((V).transpose())) + '\n')
output.print_complex_array('ref.data', _a.flatten())
print 'Ref data generated!'

