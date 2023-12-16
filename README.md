# Compositional Inductive Invariant Based Verification of NNCS

### Dependency
- Z3 ver: 4.8.17
- Python Packages: see `requirements.txt`
  - PyTorch 1.11.0 with CUDA 11.3
  - `AutoLiRPA` is required only for compositional method.
- GNU `time` for timing
- `make`

### Reproduce the Experiments
Refer to `Makefile`. Use `make [opts]` to run the experiments, available `[opts]` are
```
- all:                  run all experiments
- det_mono:             deterministic 2d maze - monolithic method (z3)
- nondet_mono:          nondeterministic 2d maze - monolithic method (z3)
- det_mono:             deterministic 2d maze - compositional method
- nondet_mono:          nondeterministic 2d maze - compositional method
- gen_det_mono_smt:     encoding det 2d maze checking into SMT files
- gen_nondet_mono_smt:  encoding non-det 2d maze checking into SMT files
- gen_all_mono_smt:     encoding both 2d mazes
- test_pt2z3:           Testing correctness of PyTorch to SMT encoding
- test_cex:             Testing the falsifying state predicate capturing counterexamples
- test_all:             Testing all above
```
