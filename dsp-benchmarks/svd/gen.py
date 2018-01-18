import numpy, cmath, sys, imp
output = imp.load_source('output', '../common/output.py')

numpy.set_printoptions(precision = 4, suppress = True, threshold = 1000, linewidth = 200)

N = int(sys.argv[1])

_a = numpy.random.rand(N, N) + 1j * numpy.random.rand(N, N)
a = _a.copy()
output.print_complex_array('input.data', a.flatten())
output.print_complex_array('ref.data', a.flatten())

ans = numpy.linalg.svd(a, compute_uv = False)

V = numpy.identity(N, dtype = 'complex128')

def household(v):
    if numpy.linalg.norm(v) < 1e-5:
        return v[0], numpy.ones((len(v)), dtype = 'complex128') / cmath.sqrt(len(v))
    hv = v.copy()
    alpha = cmath.exp(1j * cmath.phase(hv[0])) * numpy.linalg.norm(hv)
    hv[0] += alpha
    hv /= numpy.linalg.norm(hv)
    return alpha, hv

f,d = [],[]
r = a.copy()

for i in range(N - 1):
    alpha, hv = household(r[:,0].copy())
    r = r[:,1:] - 2 * numpy.outer(hv, numpy.dot(numpy.conj(hv), r[:,1:]))
    d.append(-alpha)
    if i != N - 2:
        alpha, hv = household(r[0,:].copy())
        r = r[1:,:] - 2 * numpy.outer(numpy.dot(r[1:,:], numpy.conj(hv)), hv)
        V[i+1:,:] = V[i+1:,:] - 2 * numpy.outer(numpy.conj(hv), numpy.dot(hv, V[i+1:,:]))
        f.append(-alpha)


d.append(r[1,0])
f.append(r[0,0])

d = numpy.array(d)
f = numpy.array(f)

TOTAL = 0

def implicit_kernel(d, f, V):
    global TOTAL
    TOTAL += 1
    n = len(d)
    assert n > 1

    mu = d[-1].conjugate() * d[-1]
    # unroll these bunch!
    alpha, hv = household(numpy.array([d[0] * d[0].conjugate() - mu, d[0] * f[0].conjugate()]))
    m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    d[0], f[0], extra, d[1] = d[0] * m[0,0] + f[0] * m[1,0], d[0] * m[0,1] + f[0] * m[1,1], d[1] * m[1,0], d[1] * m[1,1]
    V[:2,:] = numpy.dot(m, V[:2,:])

    alpha, hv = household(numpy.array([d[0],extra]))
    m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    d[0] = -alpha
    f[0], d[1] = m[0,0] * f[0] + m[0,1] * d[1], m[1,0] * f[0] + m[1,1] * d[1]
    if n != 2:
        extra = m[0, 1] * f[1]
        f[1]  = m[1, 1] * f[1]

    for i in range(1, n - 1):
        alpha, hv = household(numpy.array([f[i-1].conjugate(), extra.conjugate()]))
        m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
        f[i-1] = -alpha.conjugate()
        d[i], f[i] = d[i] * m[0,0] + f[i] * m[1,0], d[i] * m[0,1] + f[i] * m[1,1]
        extra  = d[i+1] * m[1,0]
        d[i+1] = d[i+1] * m[1,1]
        V[i:i+2,:] = numpy.dot(m, V[i:i+2,:])

        alpha, hv = household(numpy.array([d[i],extra]))
        m = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))

        d[i]   = -alpha
        f[i], d[i+1] = m[0,0] * f[i] + m[0,1] * d[i+1], m[1,0] * f[i] + m[1,1] * d[i+1]
        if i != n - 2:
            extra  = m[0, 1] * f[i+1]
            f[i+1] = m[1, 1] * f[i+1]


while True:
    i = 0
    called = False
    while i < N - 1:
        j = i
        while j < N - 1 and abs(f[j]) > 1e-5:
            j += 1
        if i != j:
            implicit_kernel(d[i:j+1], f[i:j], V[i:j+1,:])
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
sv = d
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

sigma = numpy.zeros((N, N), dtype = 'complex128')
for i in range(N):
    sigma[i, i] = sv[i]

try:
    numpy.testing.assert_allclose(
        numpy.dot(numpy.dot(U, sigma), V),
        _a,
        atol = 1e-4, rtol = 1e-4
    )
except:
    print 'WARN: Precision loss too much!'

print 'AVG ERROR: %.6f' % (abs(numpy.dot(numpy.dot(U, sigma), V) - _a).sum() / (N * N))

