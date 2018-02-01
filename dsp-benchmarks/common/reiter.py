import struct, sys

def reiterpret(n):
    pack = struct.pack('Q', n)
    return struct.unpack('ff', pack)

if __name__ == '__main__':
    # If it is run by itself, parse the data
    for line in file('raw', 'r').readlines():
        if 'inputs:' in line:
            print 'Input:',
            for j in line[8:-3].split(', '):
                print '(%.5f %.5f)' % reiterpret(int(j, 16)), 
            print
        elif 'output:' in line:
            print 'Output:',
            print ('(%.5f %.5f)') % reiterpret(int(line[7:], 16))
        else:
            elems = line.split()
            if not elems:
                continue
            print elems[0],
            for j in elems[1:]:
                try:
                    print ('(%.5f %.5f)') % reiterpret(int(j, 16)),
                except:
                    print j,
            print

