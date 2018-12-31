import sys
import random

try:
    m, ratio = int(sys.argv[1]), float(sys.argv[2])
except:
    m, ratio = int(sys.argv[1]), float(sys.argv[2])

with open('dense_activations.data', 'w') as f_input:
    nnz = int(m*ratio)
    line = []
    prev_ind = 0
    for i in range(nnz):
        #generate ith non-zero value
        upper = int(i/ratio)
        if(prev_ind!=upper):
            ind = random.randint(prev_ind+1, int(i/ratio))
        else:
            ind=prev_ind
        for j in range(prev_ind+1, ind):
            line.append('%d %d\n' % ( j, 0))
        prev_ind = ind
        line.append('%d %d\n' % ( ind , random.randint(1,500)))
    f_input.write(''.join(line))
