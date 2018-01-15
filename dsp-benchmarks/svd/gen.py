import numpy, math, cmath

numpy.set_printoptions(precision = 6, suppress = True, threshold = 1000, linewidth = 200)

N = 6

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


print ans

last = -1

for TOTAL in range(1000):
    for i in range(N - 1):
        if i:
            m = household(a[i-1,i].conjugate(), a[i-1,i+1].conjugate())
            a[i-1:,i:i+2] = numpy.dot(a[i-1:,i:i+2], m)
        else:
            mu = a[last, last] * a[last, last].conjugate()
            print mu
            m = household(a[0, 0] * a[0, 0].conjugate() - mu, a[0, 0] * a[0, 1].conjugate())
            a[:,0:2] = numpy.dot(a[:,0:2], m)
        a[i:i+2,i:] = numpy.dot(household(a[i,i],a[i+1,i]), a[i:i+2,i:])
    while last > -N and abs(a[last - 1, last]) < 1e-5:
        last -= 1
    if last <= -N:
        break
    ata = numpy.dot(numpy.conj(a.transpose()), a)
    print ata
    print numpy.sqrt(numpy.diag(ata))
    raw_input()

print TOTAL
a = numpy.diag(a)
a = numpy.sqrt(a * numpy.conj(a))
a = numpy.array(map(lambda x: x.real, a))
print a
print ans
print abs(ans - a)

