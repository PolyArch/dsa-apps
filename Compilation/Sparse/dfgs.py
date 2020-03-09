#/usr/bin/env python3

exec(open('../Common/dfgs.py').read())

spec = {
    # Make this 5 so that only unroll factor 1 is legal.
    'join.cc': 5,
    'hist.cc': 5,
}

run_spec(spec)
