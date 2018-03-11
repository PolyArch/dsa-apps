n = input()
init = 24 * n
compute = sum((n - i) * (n - i - 1) / 2 for i in xrange(n))
print init + compute
