#!/usr/bin/env python3
import os, re

def run_shell(s):
    return os.popen(s + ' > /dev/null 2>&1').read()

def run_sb(env = '', sb = 'origin'):
    run_shell('%s make sb-%s.log' % (env, sb))


def find_line(s, raw):
    return [i for i in raw if s in i]

def find_line_val(s, raw):
    if s[-1] != ':':
        print(s)
        raise Exception('Cannot find value without :')
    tmp = find_line(s, raw)
    if len(tmp) != 1:
        print(tmp)
        raise Exception('More than one line with the same prefix')
    return float(tmp[0].strip(s).split()[0])
    

def analyze_log(s):
    raw = open(s, 'r').readlines()

    if s == 'ooo.log' or s == 'physical.log':
        for i in raw:
            if 'ticks:' in i:
                return [int(i.split(':')[1])]

    res = []
    cycles = find_line_val('Cycles:', raw)
    insts  = find_line_val('Control Core Insts Issued:', raw)
    res.append('%.2f' % (cycles / 1000.))
    res.append('%d' % int(insts))

    if 'sb-' in s:
        res.append('%.2f' % (cycles / insts))
        res.append(find_line_val('CGRA Insts / Cycle:', raw))
        line = find_line('DMA_LOAD:', raw)
        line = line[0]
        line = line.strip('DMA_LOAD:')
        index = line.find('(')
        rindex= line.find('B/c')
        bandw = float(line[index + 1: rindex])
        ratio = float(line[:index]) * 100
        res.append('%.2f (%.2f%%)' % (bandw, ratio))

    return res

def analyze_breakdown(s):
    raw = open(s, 'r').readlines()
    line = find_line('Cycle Breakdown: ', raw)[0].lstrip('Cycle Breakdown: ')
    res = map(lambda x: float(x.split(':')[1]), line.strip().split())
    ticks = '%.2f' % (find_line_val('Cycles:', raw) / 1000.)
    return [ticks] + list(res)


def run_cpu(env = ''):
    #run_shell(env + ' make physical.log')
    run_shell(env + ' make ooo.log')
    res = []
    #res += analyze_log('physical.log')
    res += analyze_log('ooo.log')
    return res

def run(cases, template, softbrains):
    run_shell('make ultraclean')
    open('breakdowns.csv', 'w').write(' '.join(map(lambda x: template % x, cases)) + '\n')
    open('breakdowns.csv', 'a').write('OoO ' + ' '.join(map(lambda x: 'sb-%s' % x, softbrains)) + '\n')
    for i in cases:
        run_shell('make clean')
        line = []
        env = template % i
        print('Run ' + env)
        line += run_cpu(env)
        breakdowns = line[:]
        print('CPU Done')
        for sb in softbrains:
            run_sb(env, sb)
            line += analyze_log('sb-%s.log' % sb)
            breakdowns += analyze_breakdown('sb-%s.log' % sb)
            print(sb + ' SB Done')
        open('res.csv', 'a').write('\t'.join(map(str, line)) + '\n')
        open('breakdowns.csv', 'a').write('\t'.join(map(str, breakdowns)) + '\n')

if __name__ == '__main__':
    print('Nothing to be done yet...')

