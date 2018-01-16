import numpy, cmath, sys, imp

numpy.set_printoptions(precision = 4, suppress = True, threshold = 1000, linewidth = 200)

N = int(sys.argv[1])

a = numpy.random.rand(N, N) + 1j * numpy.random.rand(N, N)

ans = numpy.linalg.svd(a, compute_uv = False)

for i in range(N - 1):
    house = a[i:,i].copy()
    alpha = cmath.exp(1j * cmath.phase(house[0])) * numpy.linalg.norm(house)
    house[0] += alpha
    house /= numpy.linalg.norm(house)
    #make it fine-grained later
    h = numpy.identity(N - i, dtype = 'complex128') - 2 * numpy.outer(house, numpy.conj(house))
    a[i:,i:] = numpy.dot(h, a[i:,i:])

    if i != N - 2:
        house = a[i,i+1:].copy()
        house[0] += cmath.exp(1j * cmath.phase(house[0])) * numpy.linalg.norm(house)
        house /= numpy.linalg.norm(house)
        #make it fine-grained later
        h = numpy.identity(N - i - 1, dtype = 'complex128') - 2 * numpy.outer(numpy.conj(house), house)
        a[i:,i+1:] = numpy.dot(a[i:,i+1:], h)
        #print a

def household(x, y):
    v = numpy.array([x, y])
    if numpy.linalg.norm(v) < 1e-5:
        return numpy.identity(2)
    hv = v.copy()
    alpha = cmath.exp(1j * cmath.phase(hv[0])) * numpy.linalg.norm(hv)
    hv[0] += alpha
    hv /= numpy.linalg.norm(hv)
    h = numpy.identity(2, dtype = 'complex128') - 2 * numpy.outer(hv, numpy.conj(hv))
    return h

TOTAL = 0

def implicit_kernel(a):
    global TOTAL
    TOTAL += 1
    shape = a.shape
    assert shape[0] == shape[1]
    n = shape[0]
    assert n > 1
    
    t = numpy.dot(numpy.conj(a[-1:,-1:].transpose()), a[-1:,-1:])
    mu = t[-1, -1]
    m = household(a[0, 0] * a[0, 0].conjugate() - mu, a[0, 0] * a[0, 1].conjugate())
    a[:2,0:2] = numpy.dot(a[:2,0:2], m)
    a[:2,:3] = numpy.dot(household(a[0,0],a[1,0]), a[:2,:3])

    for i in range(1, n - 1):
        m = household(a[i-1,i].conjugate(), a[i-1,i+1].conjugate())
        a[i-1:i+2,i:i+2] = numpy.dot(a[i-1:i+2,i:i+2], m)
        a[i:i+2,i:i+3] = numpy.dot(household(a[i,i],a[i+1,i]), a[i:i+2,i:i+3])

while True:
    i = 0
    called = False
    while i < N - 1:
        j = i
        while j < N - 1 and abs(a[j, j + 1]) > 1e-5:
            j += 1
        if i != j:
            implicit_kernel(a[i:j+1, i:j+1])
            called = True
        i = j + 1
    if not called:
        break

print TOTAL
a = numpy.diag(a)
a = numpy.sqrt(a * numpy.conj(a))
a = numpy.array(map(lambda x: x.real, a))
print a
print ans
print abs(ans - sorted(a)[::-1])

