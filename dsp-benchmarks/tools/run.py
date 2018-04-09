#!/usr/bin/env python3
import os, re, math

def run_shell(s):
    return os.popen(s + ' > /dev/null 2>&1').read()

def run(sizes, template, softbrains, nonmkl = False, ultraclean = True):
    if ultraclean:
        run_shell('make ultraclean')
    for i in sizes:
        if isinstance(i, tuple):
            ii = '_'.join([str(j) for j in i])
        else:
            ii = str(i)
        os.mkdir('log_' + str(ii))
        run_shell('make clean')
        env = template % i

        print('Run Case %s...' % env)
        for sb in softbrains:
            run_shell('%s make sb-%s.log' % (env, sb))
            os.rename('./sb-%s.log' % sb, './log_%s/sb-%s.log' % (ii, sb))
            print('sb-%s Done' % sb)

        if not nonmkl:
            for threads in [1, 4]:
                print(os.popen('%s OMP_NUM_THREADS=%d NUM_PTHREADS=%d make mkl.exe' % (env, threads, threads)).read())
                print('MKL Built...')
                mkl_ticks = []
                for j in range(100):
                    lst = os.popen('OMP_NUM_THREADS=%d NUM_PTHREADS=%d ./mkl.exe' % (threads, threads)).read().split()
                    #print(lst)
                    for elem in lst:
                        try:
                            ticks = int(elem)
                            break
                        except:
                            pass
                    mkl_ticks.append(ticks)
                avg = sum(sorted(mkl_ticks)[:25]) / 25.
                open('./log_%s/%d-mkl.log' % (ii, threads), 'w').write('\n'.join(map(str, mkl_ticks)) + ('\nticks: %f' % avg))
                print('MKL %d-thread Run 100 Times! %f us\n' % (threads, avg))
                os.remove('./mkl.exe')
        os.rename('./gen.log', './log_%s/gen.log' % ii)

if __name__ == '__main__':
    print('Nothing to be done yet')
