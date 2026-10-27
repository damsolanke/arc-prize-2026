PY ?= python3

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make test        — pytest (54 tests)"
	@echo "  make lint        — ruff check"
	@echo "  make baseline    — run all heuristic solvers on eval + train; writes JSON"
	@echo "  make taxonomy    — compute task taxonomy; writes JSON"
	@echo "  make analyze     — per-task breakdown + oracle upper bound"
	@echo "  make tables      — regenerate paper/tables/ from latest JSON results"
	@echo "  make figures     — render all 120 eval task figures (gitignored)"
	@echo "  make reproduce   — everything: lint + test + baseline + taxonomy + tables"
	@echo "  make clean       — remove caches + gitignored artifacts"

.PHONY: test
test:
	$(PY) -m pytest -q tests/

.PHONY: lint
lint:
	$(PY) -m ruff check src/ tests/

.PHONY: baseline
baseline:
	$(PY) scripts/run_baseline.py

.PHONY: taxonomy
taxonomy:
	$(PY) scripts/compute_taxonomy.py

.PHONY: analyze
analyze:
	$(PY) scripts/per_task_analysis.py

.PHONY: tables
tables:
	$(PY) scripts/update_paper_tables.py

.PHONY: figures
figures:
	$(PY) -c "from pathlib import Path; import sys; sys.path.insert(0, '.'); \
	from src.eval.loader import EVAL_DIR, load_split; \
	from src.utils.viewer import save_task_figure; \
	out = Path('docs/images/eval_tasks'); out.mkdir(parents=True, exist_ok=True); \
	[save_task_figure(t, out / f'{t.task_id}.png') for t in load_split(EVAL_DIR)]"

.PHONY: reproduce
reproduce: lint test baseline taxonomy analyze tables
	@echo ""
	@echo "=== reproduce complete. JSON results in experiments/results/, tables in paper/tables/ ==="

.PHONY: clean
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf experiments/results
	rm -rf docs/images/eval_tasks
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned caches + gitignored artifacts"
