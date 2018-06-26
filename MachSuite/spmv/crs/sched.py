#!/usr/bin/env python3

import imp

sched = imp.load_source('run', '../../../common/sched.py')

sched.run_scheduler(['mm_lanes.dfg'])
