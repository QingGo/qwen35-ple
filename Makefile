.PHONY: help sync lint test check

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

sync: ## uv 同步依赖（含兄弟仓库 path 覆盖）
	uv sync --all-groups

CI_SCRIPTS := $(shell grep -o 'scripts/[^ ]*\.py' .github/workflows/ci.yml)

lint: ## ruff 检查（与 CI 相同范围）
	uv run ruff check src tests $(CI_SCRIPTS)

test: ## pytest 冒烟
	uv run pytest -q

check: lint test ## lint + test
