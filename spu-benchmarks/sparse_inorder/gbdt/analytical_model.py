import numpy, sys
import math

try:
    Bn, B, C = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
except:
# Bn and B are in bytes/cycle
    Bn = 16
    B = 256
    C = 64

sim_cycles = [0] * 4
# this is for M=C, this needs to take M into consideration as well

def calc_limiting_factor(Bnreq, Breq):
    if(Bnreq==0 | Breq==0):
        link_share = 0
    else:
        # link_share = min ((Bnreq + Breq) / Bnreq, (Bnreq + Breq) / Breq)
        # link_share = (Bnreq + Breq) / max(Bnreq, Breq)
        link_share = (Bnreq + Breq) / Bn
    # link_share = link_share * (max(Bnreq, Breq) / Bn)
    nw_band = max(Bnreq, Breq) / Bn
    mem_band = Breq*C / B
    return max(link_share, nw_band, mem_band)
    
def cycle_calc():
    cycles = 0
    
    # Phase 1: histogram building
    cycles = cycles + (sim_cycles[0] * calc_limiting_factor(16, 8))
    
    # Phase 2: cumsum and local reduction
    cycles = cycles + (sim_cycles[1] * calc_limiting_factor(0, 0))
    
    # Phase 3: hierarchical reduction
    cycles = cycles + math.log(C,2)*(3 + Bn/(3*8))
    
    # Phase 4: mapping
    cycles = cycles + (sim_cycles[2] * calc_limiting_factor(8, 8))
    
    # Phase 5: error calculation (I think this is like binning?)
    cycles = cycles + (sim_cycles[3] * calc_limiting_factor(8, 8))
    return cycles

lines = tuple(open('simulation_results.txt','r'))
for i in range(len(lines)):
    x = lines[i].split()
    print x
    sim_cycles[i] = int(x[1])

print "n/w bandwidth =", Bn, " memory bandwidth =", B, " number of CGRA's in system =", C
print "Cycles =", cycle_calc()
