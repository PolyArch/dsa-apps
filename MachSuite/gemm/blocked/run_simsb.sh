#! /bin/bash

SBCONFIG=$RISCV/configs/diannao_simd64.sbmodel spike  --ic=128:4:64 --dc=128:4:64 --l2=1024:8:64  --extension=softbrain $RISCV/riscv64-unknown-elf/bin/pk  ./gemm_sb input.data check.data

