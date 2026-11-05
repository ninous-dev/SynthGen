SHELL := /bin/bash

# ============================================================================
# Targets
# ============================================================================
.PHONY: help sub_mida sub_vascular sub_vascular2 sub_skull sub_bigBrain sub_allen sub_all sub_ano sub_all_ano \
        sub_mida_ano sub_vascular_ano sub_vascular2_ano sub_skull_ano sub_bigBrain_ano sub_allen_ano \
        submit submit_ano clean

help:
	@echo "Available targets:"
	@echo "  make sub_mida                       - Submit MIDA (all anos)"
	@echo "  make sub_vascular                   - Submit Vascular (all anos)"
	@echo "  make sub_vascular2                  - Submit Vascular2 (all anos)"
	@echo "  make sub_skull                      - Submit Skull (all anos)"
	@echo "  make sub_bigBrain                   - Submit BigBrain (all anos)"
	@echo "  make sub_allen                      - Submit Allen (all anos)"
	@echo "  make sub_all                        - Submit all datasets for all anos"
	@echo "  make sub_ano ANO=<ano_name>         - Submit all datasets for a specific ano"
	@echo "  make sub_mida_ano ANO=<ano_name>    - Submit MIDA for a specific ano"
	@echo "  make sub_vascular_ano ANO=<ano_name>- Submit Vascular for a specific ano"
	@echo "  make sub_vascular2_ano ANO=<ano_name>- Submit Vascular2 for a specific ano"
	@echo "  make sub_skull_ano ANO=<ano_name>   - Submit Skull for a specific ano"
	@echo "  make sub_bigBrain_ano ANO=<ano_name>- Submit BigBrain for a specific ano"
	@echo "  make sub_allen_ano ANO=<ano_name>   - Submit Allen for a specific ano"
	@echo "  make sub_all_ano                    - Submit all datasets (anos 1&3 for allen/bigBrain, anos 1&2 for others)"

# ============================================================================
# Number of jobs for each dataset
# ============================================================================
NB_JOBS_1 := 500
NB_JOBS_2 := 500
NB_JOBS_3 := 200
NB_JOBS_4 := 1000
NB_JOBS_5 := 1000
NB_JOBS_6 := 500


sub_mida:
	$(MAKE) submit DSname=mida DSnum=1 mem=100G

sub_mida_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=mida DSnum=1 mem=100G ANO=$(ANO)

sub_vascular:
	$(MAKE) submit DSname=vascular DSnum=2 mem=100G

sub_vascular_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=vascular DSnum=2 mem=100G ANO=$(ANO)

sub_vascular2:
	$(MAKE) submit DSname=vascular2 DSnum=6 mem=100G

sub_vascular2_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=vascular2 DSnum=6 mem=100G ANO=$(ANO)

sub_skull:
	$(MAKE) submit DSname=skull DSnum=3 mem=100G

sub_skull_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=skull DSnum=3 mem=100G ANO=$(ANO)

sub_bigBrain:
	$(MAKE) submit DSname=bigBrain mem=180G DSnum=4

sub_bigBrain_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=$(ANO)

sub_allen:
	$(MAKE) submit DSname=allen DSnum=5 mem=100G

sub_allen_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=$(ANO)

sub_all:
	@echo "🚀 Submitting all datasets..."
	@echo "-----------------------------"
	$(MAKE) sub_mida
	$(MAKE) sub_vascular
	$(MAKE) sub_vascular2
	$(MAKE) sub_skull
	$(MAKE) sub_bigBrain
	$(MAKE) sub_allen
	@echo "All jobs submitted"

sub_all_ano:
	@echo "🚀 Submitting all datasets with per-dataset ano sets..."
	@echo "-----------------------------"
	$(MAKE) submit_ano DSname=mida DSnum=1 mem=100G ANO=1
	$(MAKE) submit_ano DSname=mida DSnum=1 mem=100G ANO=2
	$(MAKE) submit_ano DSname=vascular DSnum=2 mem=100G ANO=1
	$(MAKE) submit_ano DSname=vascular DSnum=2 mem=100G ANO=2
	$(MAKE) submit_ano DSname=vascular2 DSnum=6 mem=100G ANO=1
	$(MAKE) submit_ano DSname=vascular2 DSnum=6 mem=100G ANO=2
	$(MAKE) submit_ano DSname=skull DSnum=3 mem=100G ANO=1
	$(MAKE) submit_ano DSname=skull DSnum=3 mem=100G ANO=2
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=3
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=4
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=3
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=4
	@echo "All jobs submitted"

sub_all_no_skull:
	@echo "🚀 Submitting all datasets with per-dataset ano sets..."
	@echo "-----------------------------"
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=3
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=4
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=3
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=4
	@echo "All jobs submitted"

sub_ano:
	@if [ -z "$(ANO)" ]; then echo "❌ Please specify ANO=<ano_name>"; exit 1; fi
	@echo "🚀 Submitting all datasets for ano: $(ANO)..."
	@echo "-----------------------------"
	$(MAKE) submit_ano DSname=mida DSnum=1 mem=100G ANO=$(ANO)
	$(MAKE) submit_ano DSname=vascular DSnum=2 mem=100G ANO=$(ANO)
	$(MAKE) submit_ano DSname=vascular2 DSnum=6 mem=100G ANO=$(ANO)
	$(MAKE) submit_ano DSname=skull DSnum=3 mem=100G ANO=$(ANO)
	$(MAKE) submit_ano DSname=bigBrain DSnum=4 mem=180G ANO=$(ANO)
	$(MAKE) submit_ano DSname=allen DSnum=5 mem=100G ANO=$(ANO)
	@echo "All jobs submitted for ano: $(ANO)"

submit_ano:
	@set -euo pipefail; \
	source env.sh; \
	echo "Submitting $(DSname) for ano $(ANO)..."; \
	echo "-----------------------------"; \
	echo "GENERATION_DIR is $$GENERATION_DIR"; \
	jobdir="$$GENERATION_DIR/$(DSname)"; \
	mkdir -p "$$jobdir/logs"; \
	names=($$ANO_NAMES); \
	i=0; \
	name="$(ANO)"; \
	for a in $$ANO_LIST; do \
		if [ "$$a" = "$(ANO)" ]; then name=$${names[$$i]}; break; fi; \
		i=$$((i+1)); \
	done; \
	njobs="$(if $(FULL),$(NB_JOBS_$(DSnum)),$$NB_JOBS)"; \
	echo "nbjobs asked: $$njobs"; \
	if [ -z "$$njobs" ]; then echo "❌ NB_JOBS introuvable pour DSnum=$(DSnum) (FULL=$(FULL))"; exit 1; fi; \
	sbatch -p "$$PARTITION" \
		--array=1-"$$njobs" \
		--cpus-per-task="$$CPU_PER_TASK" \
		-t "$$DEFAULT_TIMELIM" \
		--mem="$${mem:-$$DEFAULT_MEM}" \
		--job-name="$(DSname)_$$name" \
		-o "$$jobdir/logs/$(DSname)_$$name-%A_%a.log" \
		-e "$$jobdir/logs/$(DSname)_$$name-%A_%a.err" \
		generation_ano.sh -d "$(DSnum)" -z "$$NB_CONTRASTS" -a "$(ANO)"
	@echo "$(DSname) submitted for ano $$name (id: $(ANO))"



submit:
	@source env.sh; \
	echo "Submitting $(DSname)..."; \
	echo "-----------------------------"; \
	echo "GENERATION_DIR is $$GENERATION_DIR"; \
	jobdir="$$GENERATION_DIR/$(DSname)"; \
	mkdir -p "$$jobdir/logs"; \
	names=($$ANO_NAMES); \
	i=0; \
	for a in $$ANO_LIST; do \
		name=$${names[$$i]}; \
		sbatch -p "$$PARTITION" \
			--array=1-"$(if $(FULL),$(NB_JOBS_$(DSnum)),$$NB_JOBS)" \
			--cpus-per-task="$$CPU_PER_TASK" \
			-t "$$DEFAULT_TIMELIM" \
			--mem="$${mem:-$$DEFAULT_MEM}" \
			--job-name="$(DSname)_$$name" \
			-o "$$jobdir/logs/$(DSname)_$$name-%A_%a.log" \
			-e "$$jobdir/logs/$(DSname)_$$name-%A_%a.err" \
			generation_ano.sh -d "$(DSnum)" -z "$$NB_CONTRASTS" -a "$$a"; \
		i=$$((i+1)); \
	done; \
	echo "$(DSname) submitted"