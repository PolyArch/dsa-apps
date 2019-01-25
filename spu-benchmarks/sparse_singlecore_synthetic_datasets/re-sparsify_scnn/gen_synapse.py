import numpy, sys
import random

try:
    Kx, ratio, Ni, Tn = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

except:
    Kx, ratio, Ni, Tn = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

#duplicate for Ni synapses
#Kx*Ky*Tn weights and duplicate for Ni
def get_synapse():
    nnz = Kx*Kx*Tn*ratio
    nz = Kx*Kx*Tn*(1-ratio)
    dist_per_elem = int(nz/nnz)
    random.seed(0)
    # print nnz
    # print nz
    # print dist_per_elem
    with open('input_synapse.data', 'a+') as f_input:
        for j in range(int(nnz)):
            line = []
            line.append('%.2f %d' % (256*random.random(), dist_per_elem))
            f_input.write(' '.join(line) + '\n')

        
open('input_synapse.data', 'w').close()
#generating Tn neurons
for i in range(Ni):
    get_synapse()
