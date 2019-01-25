import numpy, sys
import random

try:
    Tx, Tn = int(sys.argv[1]), int(sys.argv[2])

except:
    Tx, Tn = int(sys.argv[1]), int(sys.argv[2])

def get_neuron():
    random.seed(0)
    with open('output_neuron.data','a+') as f_input:
        line = []
        for j in range(Tx*Tx):
            line.append('%.2f' % (random.random()*random.choice([-1,+1])))
        f_input.write(' '.join(line) + '\n')

open('output_neuron.data', 'w').close()
#generating Tn neurons
for i in range(Tn):
    get_neuron()
