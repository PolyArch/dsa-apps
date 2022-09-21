exec(open('common/extract_dfgs.py').read())


import glob

setup()

for filename in glob.iglob('*.c'):
    extract_dfg(filename, [1])

create_dfglist()