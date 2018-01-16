import numpy, cmath, sys, imp

numpy.set_printoptions(precision = 4, suppress = True, threshold = 1000, linewidth = 200)

N = int(sys.argv[1])

_a = numpy.random.rand(N, N) + 1j * numpy.random.rand(N, N)
a = _a.copy()

ans = numpy.linalg.svd(a, compute_uv = False)

V = numpy.identity(N, dtype = 'complex128')

#ata = numpy.dot(numpy.conj(a).transpose(), a)

def household(v):
    if numpy.linalg.norm(v) < 1e-5:
        return numpy.identity(2)
    hv = v.copy()
    alpha = cmath.exp(1j * cmath.phase(hv[0])) * numpy.linalg.norm(hv)
    hv[0] += alpha
    hv /= numpy.linalg.norm(hv)
    return alpha, hv
    #h = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    #return h


f,d = [],[]
r = a.copy()

for i in range(N - 1):
    alpha, hv = household(a[i:,i].copy())
    #make it fine-grained later
    h = numpy.identity(N - i, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    #a[i:,i:] = numpy.dot(h, a[i:,i:])

    a[i, i] = -alpha
    a[i:,i+1:] = numpy.dot(h, a[i:,i+1:])
    a[i+1:,i] = numpy.zeros((N-i-1,), 'complex128')
    f.append(-alpha)

    if i != N - 2:
        alpha, hv = household(a[i,i+1:].copy())
        #make it fine-grained later
        h = numpy.identity(N - i - 1, dtype = 'complex128') - 2 * numpy.outer(numpy.conj(hv), hv)
        #a[i:,i+1:] = numpy.dot(a[i:,i+1:], h)

        a[i,i+1] = -alpha
        a[i,i+2:] = numpy.zeros((N-i-2), 'complex128')
        a[i+1:,i+1:] = numpy.dot(a[i+1:,i+1:], h)
        d.append(-alpha)

        V[i+1:,:] = numpy.dot(h, V[i+1:,:])

        """ check passed!
        invsd = numpy.dot(a, V)
        numpy.testing.assert_allclose(
            numpy.dot(numpy.conj(invsd.transpose()), invsd),
            ata,
            atol = 1e-5,
            rtol = 1e-5
        )
        """

TOTAL = 0

def implicit_kernel(a):
    global TOTAL
    TOTAL += 1
    shape = a.shape
    assert shape[0] == shape[1]
    n = shape[0]
    assert n > 1

    q = numpy.identity(n, dtype = 'complex128')

    """ check pass
    ata = numpy.dot(numpy.conj(a).transpose(), a)
    """

    t = numpy.dot(numpy.conj(a[-1:,-1:].transpose()), a[-1:,-1:])
    mu = t[-1, -1]
    # unroll these bunch!
    alpha, hv = household(numpy.array([a[0, 0] * a[0, 0].conjugate() - mu, a[0, 0] * a[0, 1].conjugate()]))
    m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    q[:2,:2] = m
    a[:2,:2] = numpy.dot(a[:2,:2], m)

    alpha, hv = household(numpy.array([a[0,0],a[1,0]]))
    m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    a[:2,:3] = numpy.dot(m, a[:2,:3])


    """ check pass!
    invsd = numpy.dot(a, q)
    numpy.testing.assert_allclose(
        numpy.dot(numpy.conj(invsd.transpose()), invsd),
        ata,
        atol = 1e-5,
        rtol = 1e-5
    )
    """

    for i in range(1, n - 1):
        alpha, hv = household(numpy.array([a[i-1,i].conjugate(), a[i-1,i+1].conjugate()]))
        m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
        a[i-1:i+2,i:i+2] = numpy.dot(a[i-1:i+2,i:i+2], m)
        q[i:i+2,:i+2] = numpy.dot(m, q[i:i+2,:i+2])

        """ check pass!
        invsd = numpy.dot(a, q)
        numpy.testing.assert_allclose(
            numpy.dot(numpy.conj(invsd.transpose()), invsd),
            ata,
            atol = 1e-5,
            rtol = 1e-5
        )
        """

        alpha, hv = household(numpy.array([a[i,i],a[i+1,i]]))
        m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
        a[i:i+2,i:i+3] = numpy.dot(m, a[i:i+2,i:i+3])

    return q

while True:
    i = 0
    called = False
    while i < N - 1:
        j = i
        while j < N - 1 and abs(a[j, j + 1]) > 1e-6:
            j += 1
        if i != j:
            q = implicit_kernel(a[i:j+1, i:j+1])
            V[i:j+1,:] = numpy.dot(q, V[i:j+1,:])

            """ check pass!
            invsd = numpy.dot(a, V)
            numpy.testing.assert_allclose(
                numpy.dot(numpy.conj(invsd.transpose()), invsd),
                ata,
                atol = 1e-5,
                rtol = 1e-5
            )
            """

            called = True
        i = j + 1
    if not called:
        break

""" check pass!
invsd = numpy.dot(a, V)
numpy.testing.assert_allclose(
    numpy.dot(numpy.conj(invsd.transpose()), invsd),
    ata,
    atol = 1e-5,
    rtol = 1e-5
)
"""

print "Total iteration: %d" % TOTAL
sv = numpy.diag(a)
sv = numpy.real(numpy.sqrt(sv * numpy.conj(sv)))

try:
    numpy.testing.assert_allclose(numpy.sort(sv)[::-1], ans, atol = 1e-5, rtol = 1e-5)
    print "Check pass!"
except:
    print "ERROR: SV not computed correctly"
    print sv
    print ans
    quit()

print 'Singular value:', sv


U = numpy.dot(_a, numpy.conj(V).transpose())
for i in range(N):
    U[:,i] /= sv[i]

""" check pass!
numpy.testing.assert_allclose(
    numpy.dot(numpy.conj(U).transpose(), U),
    numpy.identity(N, dtype = 'complex128'),
    atol = 1e-5, rtol = 1e-5
)
"""

#verify code:

sigma = numpy.sqrt(numpy.dot(numpy.conj(a).transpose(), a))

try:
    numpy.testing.assert_allclose(
        numpy.dot(numpy.dot(U, sigma), V),
        _a,
        atol = 1e-4, rtol = 1e-4
    )
except:
    print 'WARN: Precision loss too much!'

print 'AVG ERROR: %.6f' % (abs(numpy.dot(numpy.dot(U, sigma), V) - _a).sum() / (N * N))

