#!/usr/bin/env python3

import imp

sched = imp.load_source('run', '../../common/sched.py')

sched.schedule_and_simulate("fft.log", "sb-new.exe", ["compute.dfg", "fine1.dfg", "fine2.dfg"], "$SS_TOOLS/configs/revel.sbmodel")
