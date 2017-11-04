import sys, numpy
from math import sin, cos, pi 
n = int(sys.argv[1])
numpy.set_printoptions(suppress = True, precision = 4., linewidth = 180, threshold = numpy.nan)

_a = numpy.random.rand(n).astype('complex64')
numpy.savetxt('input.data', _a.flatten())
print '%d-length input generated' % n
a = _a.copy()

"""
"""
def brute_force(a):
    n = len(a)
    res = []
    for i in xrange(n):
        w = cos(2 * pi * i / n) + 1j * sin(2 * pi * i / n)
        cur = 0
        for j in xrange(n):
            cur += w ** j * a[j]
        res.append(cur)
    return numpy.array(res)

def non_recursive(_a):
    a = _a.copy()
    n = len(_a)
    blocks = n / 2
    w = numpy.array([cos(2 * pi * i / n) + 1j * sin(2 * pi * i / n) for i in xrange(n / 2)])
    while blocks:
        span = n / blocks
        dup = a.copy()
        for j in xrange(0, span / 2 * blocks, blocks):
            for i in xrange(0, blocks):
                L, R = dup[2 * j + i], dup[2 * j + i + blocks]
                a[i + j] = L + w[j] * R
                a[i + j + span / 2 * blocks] = L - w[j] * R
        blocks /= 2
    return a

numpy.testing.assert_allclose(non_recursive(a), brute_force(a), atol = 1e-4)
print 'check pass!'
numpy.savetxt('ref.data', non_recursive(a).flatten())
print 'output generated!'
