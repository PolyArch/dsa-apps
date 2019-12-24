import sys
import random
import numpy as np
import scipy.stats as ss

try:
    m, ratio = int(sys.argv[1]), float(sys.argv[2])
except:
    m, ratio = int(sys.argv[1]), float(sys.argv[2])

# a single vector of len nnz
with open('input_activations.data', 'w') as f_input:
    col_nnz = int(m*ratio)
    range_min = m/2
    x = np.arange(-range_min, range_min)
    xU, xL = x + col_nnz, x
    scale_factor = m*ratio # *0.125*0.125*0.75
    prob = ss.norm.cdf(xU, scale = scale_factor) - ss.norm.cdf(xL, scale = scale_factor)
    prob = prob / prob.sum()
    print(m)
    print(np.shape(prob))
    random.seed(0)
    indices = np.random.choice(m, col_nnz, replace=False, p=prob)
    indices = sorted(indices)

    line = []
    for col_id in indices: # write a given row
        line.append('%d %d\n' % (col_id, random.randint(1,50)))
    f_input.write(''.join(line))
