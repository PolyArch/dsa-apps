import imp
output = imp.load_source('output', '../common/output.py')

# A quick POC (proof of concept) of Cholesky Decomposition in numpy.
import numpy, cmath, sys
n = int(sys.argv[1])
a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
a = a + numpy.identity(n)

output.print_complex_array('input.data', a.flatten())
print("%d x %d Input generated!" % (n, n))

Q = numpy.identity(n, dtype = 'complex128')
R = a.copy()

#print origin

for i in range(n):
    v = R[i:,i].copy()
    alpha = -cmath.exp(1j * cmath.phase(v[0])) * numpy.linalg.norm(v)
    v[0] -= alpha
    v = v / numpy.linalg.norm(v)
    H = numpy.identity(n - i, dtype = 'complex128') - 2 * numpy.outer(v, numpy.conj(v))
    temp = numpy.dot(numpy.conj(v), R[i:,i:])
    R[i:,i:] -= 2 * numpy.outer(v, temp)
    temp = numpy.dot(Q[:,i:], v)
    Q[:,i:] -= 2 * numpy.outer(temp, numpy.conj(v))

numpy.testing.assert_allclose(numpy.identity(n), numpy.dot(Q, numpy.conj(Q.transpose())), atol = 1e-5)
numpy.testing.assert_allclose(a, numpy.dot(Q, R), rtol = 1e-5)

print("Correctness check pass!")
output.print_complex_array('ref.data', a.flatten()) #numpy.concatenate((Q.flatten(), R.flatten())));
print("New data generated!")

hh = sum((n - i) + (n - i - 1) + 40 for i in range(n))
gemm_r = 0 #sum(n - i + (n - i) * (n - i) for i in range(n)) + 4 - 1
#gemm_q = sum(((n - i - 1) / 2 + 1) * n + n for i in range(n))
gemm_q = sum(((n - i - 1) + 1) * n for i in range(n))
print('ASIC Ideal: %d' % (hh + gemm_q + gemm_r))

