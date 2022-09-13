def get_unroll():
    unroll = os.getenv('unroll')
    if unroll is not None:
        unroll = int(unroll)
    else:
        unroll = 4
    return unroll

SS = os.getenv('SS')
CLEAN_UP = f'opt -disable-loop-idiom-all -load {SS}/llvm-project/build/lib/DSAPass.so -tvm-rm-bound-checker %s -o %s -S'
XFORM = f'SBCONFIG={SS}/ss-cgra-gen/IR/revel.json opt %s -load {SS}/llvm-project/build/lib/DSAPass.so -stream-specialize %s -o %s -S'
UNIFY = f'opt -load {SS}/llvm-project/build/lib/DSAPass.so -tvm-unify-signature %s -o %s -S'
