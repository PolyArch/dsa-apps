#!/usr/bin/env python3

import imp

sched = imp.load_source('run', '../../common/sched.py')

sched.schedule_and_simulate("fir.log", "sb-new.exe", ["compute.dfg"], "$SS_TOOLS/configs/revel.sbmodel")
