import os
import shutil
import glob

''' A few notes about this script.

It will automatically extract the dfgs from all c programs in the folder it is run in. The names of the dfgs will be the same as the name of the file that created them. It will also create a dfgs.list file that can be used for the scheduler.

A few notes: the generated dfg files must not have a '_' in their name. Vectorization degree is controlled by the vectorization_degree list, with associated vectorization degree options. It will be used as the U environment variable.

'''


def setup():
    dfg_dir = 'dfgs'
    if os.path.exists(dfg_dir):
        shutil.rmtree(dfg_dir)
    os.makedirs('dfgs')

def extract_dfg(filename, vectorization_degrees=[1]):
    # Get Prefix for the file
    file_prefix = filename[:-2]
    
    # Create a directory for the file
    os.makedirs('dfgs/' + file_prefix)

    for degree in vectorization_degrees:
        # Run the command to extract the DFG
        command = 'U=' + str(degree) + ' EXTRACT=1 COMPAT_ADG=0 GRANULARITY=64 make ss-' + file_prefix + '.ll'
        os.system(command)

        # Move the DFG to the directory
        for dfg_file in glob.glob('*.dfg'):
            if os.path.exists('dfgs/' + file_prefix + '/' + dfg_file):
                os.remove('dfgs/' + file_prefix + '/' + dfg_file)
            shutil.move(dfg_file, 'dfgs/' + file_prefix)
        
        # Delete excess files created
        for excess_file in glob.glob('*.ll'):
            os.remove(excess_file)

def create_dfglist():
    dfg_files = glob.glob('dfgs/*/*.dfg')
    dfg_files.sort()
    with open('dfgs.list', 'w') as f:
        prev_file = ''
        for dfg_file in dfg_files:
            if prev_file != dfg_file.split('_')[0]:
                f.write('%%' + os.linesep)
            f.write(dfg_file + os.linesep)
            prev_file = dfg_file.split('_')[0]

setup()

for filename in glob.iglob('*.c'):
    extract_dfg(filename, [1, 2, 4, 8])

create_dfglist()

