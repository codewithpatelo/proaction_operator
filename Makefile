# Pro-Action Γ Experiment Makefile

.PHONY: help install test smoke calibration preflight benchmark analyze docker

help:
	@echo "Pro-Action Γ Experiment"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  smoke        - Run smoke tests (free, no LLM)"
	@echo "  calibration  - Run calibration phase"
	@echo "  preflight    - Run preflight (1 cell, ~$0.10)"
	@echo "  benchmark    - Run full benchmark (~$15-20)"
	@echo "  analyze      - Analyze results and generate report"
	@echo "  docker       - Build Docker image"
	@echo "  clean        - Clean checkpoints and results"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

smoke:
	@echo "[SMOKE TESTS] Running free tests..."
	python tests/test_seeds.py
	python tests/test_ipd_payoff.py
	python tests/test_opponent_policies.py
	python exp/verify_hrrl_fidelity.py

calibration:
	python -m exp.runner --calibration

preflight:
	python -m exp.runner --preflight

benchmark:
	python -m exp.runner --benchmark

analyze:
	python -m exp.analyze

docker:
	docker build -t proaction-gamma .

docker-run:
	docker run --rm -it \
		--env-file .env \
		-v $(PWD)/results:/work/results \
		-v $(PWD)/checkpoints:/work/checkpoints \
		proaction-gamma --benchmark

watchdog:
	python -m exp.watchdog --worker-cmd python -m exp.runner --benchmark

clean:
	rm -rf checkpoints/* results/* reports/*
	find . -type f -name "*.tmp" -delete
	find . -type f -name "*.log" -delete

# Safety: require explicit confirmation for expensive operations
benchmark-confirm:
	@echo "WARNING: This will spend ~$15-20 in API costs"
	@echo "Type 'yes' to continue: "
	@read CONFIRM && [ "$$CONFIRM" = "yes" ] && $(MAKE) benchmark || echo "Aborted"
