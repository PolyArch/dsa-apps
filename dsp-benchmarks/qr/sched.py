#!/usr/bin/env python3

import imp

sched = imp.load_source('run', '../../common/sched.py')

sched.schedule_and_simulate("qr.log", "sb-new.exe", ["multi.dfg", "fused1.dfg", "fused2.dfg"], "$SS_TOOLS/configs/revel.sbmodel")
