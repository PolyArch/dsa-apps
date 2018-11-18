#import numpy, 
import sys
import random

# m is number of features: 4? n*ratio is number of inst
try:
    n, m, ratio, dep_ratio = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
except:
    n, m, ratio, dep_ratio = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

with open('input.data', 'w') as f_input:
    for i in range(int(n*ratio)):
        random.seed(i)
        line = []
        for j in range(int(m)):
            line.append('%d:%d ' % ( random.randint(int((i)/ratio),
                        int((i+1)/ratio-1)), random.randint(1,64)))
        f_input.write(' '.join(line) + '\n')

with open('labels.data', 'w') as f_input:
    for i in range(int(n)):
        f_input.write('%.2f %.2f\n' % (random.random(), random.random()))

with open('inst_id.data', 'w') as f_input:
    for i in range(int(n*dep_ratio)):
        f_input.write(str(random.randint(int(i/dep_ratio), int((i+1)/dep_ratio))) + '\n')
