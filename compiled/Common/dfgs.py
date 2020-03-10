#/usr/bin/env python3

import os, subprocess, shutil

def all_files(suff, path='.'):
    return [i for i in os.listdir(path) if i.endswith(suff)]

def run_spec(spec: dict):

    shutil.rmtree('dfgs', ignore_errors=True)
    os.makedirs('dfgs')

    for cc in all_files('.cc'):
        to_enum = spec.get(cc, [['U', 1, 2, 4, 8]])
        no_ext = cc[:-3]
        if isinstance(to_enum, int):
            n = to_enum
            if n == 1:
                unroll_factors = [1, 2, 4, 8]
            elif n == 2:
                unroll_factors = [1, 2, 4]
            elif n <= 4:
                unroll_factors = [1, 2]
            else:
                unroll_factors = [1]
            to_enum = []
            for i in range(n):
                to_enum.append([f'U{i+1}' if n > 1 else 'U'] + unroll_factors)
        print(cc)
        print(to_enum)

        def dfs(space, point, signature):
            if not space:
                subprocess.check_output(['make', 'clean'])
                subprocess.check_output(f'{point} EXTRACT=1 make ss-{no_ext}.bc', shell=True)
                new_name = '-'.join(map(str, signature)) + '.dfg'
                for i in all_files('.dfg'):
                    try:
                        os.mkdir(f'dfgs/{i}')
                    except:
                        pass
                    shutil.move(i, f'dfgs/{i}/{new_name}')
            else:
                u = space[0][0]
                lst = space[0][1:][:]
                while lst:
                    dfs(space[1:], f'{u}={lst[0]} {point}', signature + [lst[0]])
                    lst = lst[1:]

        dfs(to_enum, "", [])

    with open('dfgs.list', 'w') as f:
        for folder in all_files('.dfg', 'dfgs'):
            f.write('%%\n')
            for dfg in all_files('.dfg', f'dfgs/{folder}'):
                f.write(f'dfgs/{folder}/{dfg}\n')
