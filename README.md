# Compositional Inductive Invariant Based Verification of NNCS

### Docker Image For Reproducing Results
We **recommend** using Docker to reproduce the experimental results. 
1. Please make sure Docker is installed on your machine, and nvidia drivers are ready
2. Clone this repo, then go to the project root
3. Build the image by `docker build -t comp-indinv-verif-nncs:current . `
4. Run Docker by `docker run -it --runtime=nvidia --gpus all comp-indinv-verif-nncs:current` (you may need `sudo`)
5. Using `make [opts]` to run experiments (See section **Run Experiments**)
6. (Optional) Don't forget to remove and prune the image, cache, and else


### Dependencies 
- Z3 ver: 4.8.17
- Python Packages: see `requirements.txt`
  - PyTorch 1.11.0 with CUDA 11.3
  - `AutoLiRPA` is required only for compositional method.
- GNU `time` for timing
- `make`

### Run Experiements
Refer to `Makefile`. Use `make [opts]` to select and run the experiments, available `[opts]` are
```
- all:                  run all experiments
- all_mono:             run all monolithic method case studies
- all_comp:             run all compositional method case studies
- det_mono:             deterministic 2d maze - monolithic method (z3)
- nondet_mono:          nondeterministic 2d maze - monolithic method (z3)
- det_comp:             deterministic 2d maze - compositional method
- nondet_comp:          nondeterministic 2d maze - compositional method
- gen_det_mono_smt:     encoding det 2d maze checking into SMT files
- gen_nondet_mono_smt:  encoding non-det 2d maze checking into SMT files
- gen_all_mono_smt:     encoding both 2d mazes
- test_pt2z3:           Testing correctness of PyTorch to SMT encoding
- test_cex:             Testing the falsifying state predicate capturing counterexamples
- test_all:             Testing all above
```
