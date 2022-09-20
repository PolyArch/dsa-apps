import sys
sys.path.insert(1, 'common')

from extract_dfgs import setup, extract_dfg, create_dfglist
import glob

''' A few notes about this script.

It will automatically extract the dfgs from all c programs in the folder it is run in. The names of the dfgs will be the same as the name of the file that created them. It will also create a dfgs.list file that can be used for the scheduler.

A few notes: the generated dfg files must not have a '_' in their name. Vectorization degree is controlled by the vectorization_degree list, with associated vectorization degree options. It will be used as the U environment variable.

'''

setup()

for filename in glob.iglob('*.c'):
    extract_dfg(filename, [1])

create_dfglist()
