export SBCONFIG=$SS_TOOLS/configs/diannao_simd64.sbmodel
spike --ic=64:4:64 --dc=64:4:64 --l2=1024:8:64 --extension=softbrain $SS_TOOLS/riscv64-unknown-elf/bin/pk dotp
#spike --extension=softbrain $SS_TOOLS/riscv64-unknown-elf/bin/pk dotp 

