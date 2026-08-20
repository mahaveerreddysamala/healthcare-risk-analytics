.PHONY: test coverage

test:
	pytest -q

coverage:
	pytest -q --cov=src --cov-report=term-missing
