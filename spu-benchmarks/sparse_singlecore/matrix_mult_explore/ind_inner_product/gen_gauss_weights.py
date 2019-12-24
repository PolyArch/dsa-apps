import sys
import random
import numpy as np
import scipy.stats as ss

try:
    n, m, ratio = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
except:                                              
    n, m, ratio = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])

# n rows, select (m*ratio) indices for selecting the columns
with open('input_weights.data', 'w') as f_input:
    col_nnz = int(m*ratio) # num of non-zero in each row
    range_min = m/2
    x = np.arange(-range_min, range_min)
    xU, xL = x + col_nnz, x
    scale_factor = m*ratio # *0.125*0.125*0.75
    prob = ss.norm.cdf(xU, scale = scale_factor) - ss.norm.cdf(xL, scale = scale_factor)
    prob = prob / prob.sum()

    for row_id in range(n):
        random.seed(row_id)
        indices_at_row = np.random.choice(m, col_nnz, replace=False, p=prob)
        indices_at_row = sorted(indices_at_row)
        line = [] # copy final row
        for col_id in indices_at_row: # write a given row
            line.append('%d %d %d\n' % (row_id, col_id, random.randint(1,50)))
        f_input.write(''.join(line))
