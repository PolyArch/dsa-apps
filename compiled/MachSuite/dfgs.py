#/usr/bin/env python3

exec(open('../Common/dfgs.py').read())

spec = {
    # 'crs.cc': 5,
    # 'ellpack.cc': 1,
    'gemm.cc': 1,
    # 'md.cc': 1,
    # 'stencil-2d.cc': 1,
    # 'stencil-3d.cc': 1,
}

run_spec(spec)
