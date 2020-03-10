# Compiling C Code to Stream Decoupled Accelerator

## Directories

* Tests: Unit tests that verifies the functionality of each module.
* MachSuite: Workloads from [this paper](http://people.seas.harvard.edu/~reagen/papers/machsuite.pdf).
* [PolyBench](https://github.com/bollu/polybench-c): Workloads that tests polyhedral capability
* DSP: Workloads from [this paper](https://arxiv.org/abs/1905.06238).

## Run Examples

All the stream-specialized infrastructure (RISCV binary linker, spatial compiler, and hardware simulator)
should be compiled to run the workloads. Refer [this repo](https://github.com/PolyArch/ss-stack) for more
more details.

To run the workloads in each directory. Some knowledge on the compilation pipeline is required.
Say we have a source file named `a.cc` in any directories mentioned above.
This file will undergo this process to run on compilation:

1. `a.cc` -> `a.bc`: This file will be parsed by our customized `clang` that understands ss pragmas.
The ss pragmas will be encoded in scope intrinsics and loop metadata to hint the compiler for further
transformation.
``` sh
$ make a.bc
$ llvm-dis < a.bc # You can see the LLVM with metadata hint.
```
2. `a.bc` -> `ss-a.bc`, `*.dfgs`: We implement a ss-pass to transform code to ready-to-codegen LLVM IR.
It extracts and compile spatial dataflows, erases operations offloaded to the accelerator,
and encode affined the data access scalars in stream intrinsics.
``` sh
$ make ss-a.bc
$ llvm-dis < ss-a.bc # You can see the transformed LLVM IR.
```
3. `ss-a.bc` -> `opt-ss-a.bc`: The transformed IR can still apply some generic optimizations, like DSE,
CSE, and etc.
4. `opt-ss-a.bc` -> `opt-ss-a.s`: The `llc` will compile it to RISCV assembly code.
5. `opt-ss-a.s` -> `opt-ss-a.out`: The `gnu-riscv-xxx-gcc` links the assembly code to executable binaries.
``` sh
$ ./run.sh opt-ss-a.out # Instead of typing `make`, using the `run.sh` script directly run it in the simulator.
```

## Hardware DSE

If environment variable `EXTRACT=1` is specified when compilation, the pipeline will only extract the spatial
dataflows without transforming the offloaded portion (`ss-a.bc` is the same as `a.bc`). This is useful to
collect spatial dataflows to explore what is the most proper parameter of the hardware giving a set of workloads.
Our infrastructure provides this capability.

1. `cd DIR && ./dfgs.py` to collect all the spatial dataflows for each workloads with different degree of unrolling.

2. The `./dfgs.py` generates a directory `dfgs` and a file `dfgs.list`. All the collected spatial dataflows are in the
directory and the roster of the dataflow files are in `dfgs.list`.

3. Then you can use our spatial compiler to do this exploration, all the results are dumped in `viz` directory.
``` sh
$ ss_sched -f -v $SS_TOOLS/configs/initial-dse.sbmodel dfgs.list --max-iter=50
```
