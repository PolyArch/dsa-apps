import os, sys

def run_schedule(dfg, fifo, algo, model):
    if algo == 'prior':
        cmd = 'sb_sched --max-iters=1 -d %d --verbose -G %s %s' % (fifo, model, dfg)
    elif algo[0] == 's':
        cmd = 'sb_sched --algorithm gams --sub-alg %s -d %d --verbose -G --mipstart %s %s' % (algo[1:].replace('\'', '\\\''), fifo, model, dfg)
    else:
        cmd = 'sb_sched --algorithm gams --sub-alg %s -d %d --verbose -G %s %s' % (algo.replace('\'', '\\\''), fifo, model, dfg)
    print(cmd)
    return os.popen(cmd)
        
 

def extract_sched_and_mis(raw):
    sched_time = "sched_time: "
    sched_fail = "Scheduling Failed!"
    sched_miss = "latency mismatch: "
    sched_miss = "latency mismatch: "
    sched_latc = "latency: "
    calc_mis   = False
    mis        = 0
    sch        = 0
    lat        = 0
    for i in raw.split('\n'):
        if i.startswith(sched_fail):
            return None
        if i.startswith(sched_time):
            sch = float(i[len(sched_time):].rstrip().rstrip('seconds'))
        if i.startswith(sched_miss):
            mis = int(i[len(sched_miss):].rstrip())
        if i.startswith(sched_latc):
            lat = int(i[len(sched_latc):].rstrip())
    return (sch, mis, lat)

def schedule_and_simulate(
    log_file,
    exe, dfgs,
    model,
    algs = ["prior", "M.RT", "M.R.T", "MR.T", "MR.RT", "MRT", "sMRT", "sMR'.RT", "sMR.RT", "sMRT'"],
    deps = [15, 7, 3, 2, 1]
):
    for sa in algs:
        last_failure = {}
        for depth in deps:
            schedule_perf = []
            for dfg in dfgs:
                if last_failure.get(dfg, None) is None:
                    print('Schedule %s with FIFO depth %d, using sub-algorithm %s' % (dfg, depth, sa))
                    proc = run_schedule(dfg, depth, sa, model)
                    tup = extract_sched_and_mis(proc.read())
                    proc_ret = proc.close()
                    print("Scheduler done!\n")
                    if proc_ret is not None:
                        tup = None
                        last_failure[dfg] = depth
                else:
                    print("Failed @%d, skip this scheduling!" % last_failure[dfg])
                    tup = None
                schedule_perf.append(tup)

            if None not in schedule_perf:
                env = "FU_FIFO_LEN=%d SBCONFIG=%s " % (depth, model)
                cmd = '%s make %s' % (env, exe)
                print(cmd)
                proc = os.popen(cmd)
                print(proc.read())
                if proc.close() is not None:
                    cyc = 'CE'
                    print('Compilation error!')
                else:
                    print("Executable made!")
                    cmd = "%s gem5.opt ~/ss-stack/gem5/configs/example/se.py --cpu-type=minor --l1d_size=2048kB --l1d_assoc=8 --l1i_size=16kB --l2_size=16384kB --caches --cmd=./%s"
                    cmd = cmd % (env, exe)
                    print(cmd)
                    proc = os.popen(cmd)
                    lines = proc.read().split('\n')
                    print('\n'.join(lines))
                    if proc.close() is None:
                        cyc = "Cycle not found!"
                        for line in lines:
                            if line.startswith("Cycles: "):
                                print(int(line[len("Cycles: "):]))
                                cyc = line[len("Cycles: "):]
                                break
                    else:
                        cyc = "Simulator RE"
                    os.remove(exe)
            else:
                cyc = "Schedule Failure"
            log = exe + "\t" + sa + "\t" + str(depth)
            schedule_perf = map(lambda x: (1200.0, 0.0, 0) if x is None else (min(x[0], 1200.0), float(depth) / (depth + x[1]), x[2]), schedule_perf)
            for dfg, perf in zip(dfgs, schedule_perf):
                sch, mis, lat = perf
                log = log + ("\t%s\t%f\t%f\t%d" % (dfg, sch, mis, lat))
            log = log + "\t" + cyc
            print(log)
            open(log_file, 'a').write(log + "\n")

