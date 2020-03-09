#/usr/bin/env python3

exec(open('../Common/dfgs.py').read())

spec = {
    'cholesky.cc': 2,
    'fft.cc': 1,
    'mm.cc': 1,
    'qr_q.cc': 4,
    'qr_r.cc': 2,
}

run_spec(spec)
