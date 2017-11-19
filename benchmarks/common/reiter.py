import struct, sys

def reiterpret(n):
    pack = struct.pack('Q', n)
    return struct.unpack('ff', pack)

for line in file('raw', 'r').readlines():
    if 'inputs:' in line:
        print 'Input:',
        for j in line[8:-3].split(', '):
            print '(%.5f %.5f)' % reiterpret(int(j, 16)), 
        print
    elif 'output:' in line:
        print 'Output:',
        print ('(%.5f %.5f)') % reiterpret(int(line[7:], 16))
        print

#n = int(sys.argv[1], 16)
#pack = struct.pack('Q', n)
#print '(%.5f, %.5f)' % struct.unpack('ff', pack)
