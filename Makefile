# python bin
PYTHON = python3

# time bin, please use GNU time if you are using macOS
TIME = /usr/bin/time

# current directory
CUR_DIR = $(shell pwd)

ASSETS_DIR = $(CUR_DIR)/assets
DET_MAZE_DIR = $(ASSETS_DIR)/det-maze
NONDET_MAZE_DIR = $(ASSETS_DIR)/non-det-maze

DET_SMT_QUERIES_DIR = $(DET_MAZE_DIR)/mono-smt-queries
DET_MONO_LOG_DIR = $(DET_MAZE_DIR)/mono-logs

NONDET_SMT_QUERIES_DIR = $(NONDET_MAZE_DIR)/mono-smt-queries
NONDET_MONO_LOG_DIR = $(NONDET_MAZE_DIR)/mono-logs

DET_NN_DIR = $(DET_MAZE_DIR)/nn-models
NONDET_NN_DIR = $(NONDET_MAZE_DIR)/nn-models

ALL_SMT_FILES_DET = $(shell ls $(DET_SMT_QUERIES_DIR)/*.smt2 | sort -g)
ALL_SMT_FILES_NONDET = $(shell ls $(NONDET_SMT_QUERIES_DIR)/*.smt2 | sort -g)


.DEFAULT_GOAL := all

# all targets
all: det_comp nondet_comp det_mono nondet_mono 

# target: deterministic 2d maze - monolithic
det_mono:
	for f in $(shell ls $(DET_SMT_QUERIES_DIR)/*.smt2 | sort -V)/; do \
		echo "\nChecking with Z3:"; \
		echo $$(basename $$f); \
		$(TIME) -f "%e" -v z3 $$f -st > $(DET_MONO_LOG_DIR)/$$(basename $$f).log; \
	done

# target: non-deterministic 2d maze - monolithic
nondet_mono:
	for f in $(shell ls $(NONDET_SMT_QUERIES_DIR)/*.smt2 | sort -V)/; do \
		echo "\nChecking with Z3:"; \
		echo $$(basename $$f); \
		$(TIME) -f "%e" -v z3 $$f -st > $(NONDET_MONO_LOG_DIR)/$$(basename $$f).log; \
	done

# target: deterministic 2d maze - compositional
det_comp:
	$(PYTHON) $(CUR_DIR)/comp_exp.py --nn-dir $(DET_NN_DIR) -e det -ms 1000

# target: non-deterministic 2d maze - compositional
nondet_comp:
	$(PYTHON) $(CUR_DIR)/comp_exp.py --nn-dir $(NONDET_NN_DIR) -e non-det -ms 1000

# target: regenerate mono smt queires - deterministic 2d maze
gen_det_mono_smt:
	$(PYTHON) $(CUR_DIR)/gen_mono_smt_queries.py det

# target: regenerate mono smt queires - non-deterministic 2d maze
gen_nondet_mono_smt:
	$(PYTHON) $(CUR_DIR)/gen_mono_smt_queries.py non-det

gen_all_mono_smt: gen_det_mono_smt gen_nondet_mono_smt

# target: random test - correctness of coverting `pytorch fc` to `z3 smt`
test_pt2z3:
	$(PYTHON) $(CUR_DIR)/test.py pt_to_z3

# check the falsifying state predicate correctly captured counterexamples
test_cex:
	$(PYTHON) $(CUR_DIR)/test.py indeed_counterexample

test_all: test_pt2z3 test_cex