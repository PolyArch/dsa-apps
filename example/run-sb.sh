export SBCONFIG=$RISCV/configs/diannao_simd64.sbmodel
spike --ic=64:4:64 --dc=64:4:64 --l2=1024:8:64 --extension=softbrain $RISCV/riscv64-unknown-elf/bin/pk dotp
#spike --extension=softbrain $RISCV/riscv64-unknown-elf/bin/pk dotp 

