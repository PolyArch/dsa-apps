import os

for path, folder, files in os.walk('.'):
    for dfg in filter(lambda x: x.endswith('.dfg'), files):
        name = dfg.rstrip('.dfg')
        for cc in filter(lambda x: x.endswith('.cc') or x == 'Makefile', files):
            cmd = 'sed "s/%s\\.h/%s\\.dfg\\.h/g" -i %s/%s' % (name, name, path, cc)
            print cmd
            os.popen(cmd)
