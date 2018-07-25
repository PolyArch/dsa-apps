import numpy, sys
import random

try:
    Tx, ratio, rle_width, Ni = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

except:
    Tx, ratio, rle_width, Ni = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

#duplicate for Ni neurons
#n*n=Tx*Ty neuron
def get_neuron():
    nnz = Tx*Tx*ratio
    nz = Tx*Tx*(1-ratio)
    dist_per_elem = (nz/nnz)
    padding = int(nnz%rle_width)
    random.seed(0)
    # print nnz
    # print nz
    # print dist_per_elem
    with open('input_neuron.data', 'a+') as f_input:
        for j in range(int(nnz/rle_width)): #last should be padding
            line = []
            line.append('%.2f %d %.2f %d %.2f %d %.2f %d' % (256*random.random(),
                dist_per_elem*4, 256*random.random(), dist_per_elem*4,
                256*random.random(), dist_per_elem*4, 256*random.random(), dist_per_elem*4))
            f_input.write(' '.join(line) + '\n')

        line = []
        if(padding!=0):
            for j in range(padding):
                line.append('%.2f %d' %(256*random.random(), dist_per_elem*4))

            for j in range(4-padding):
                line.append('%.2f %d' %(0, 0))

            f_input.write(' '.join(line) + '\n')


open('input_neuron.data', 'w').close()
#generating Tn neurons
for i in range(Ni):
    get_neuron()
