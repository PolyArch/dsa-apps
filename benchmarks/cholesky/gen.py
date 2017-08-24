import numpy, sys

n = int(sys.argv[1])
a = numpy.random.rand(n, n)
a = (numpy.dot(a, a.transpose()))
a = a + numpy.identity(n)

numpy.savetxt('input.data', a.flatten())
