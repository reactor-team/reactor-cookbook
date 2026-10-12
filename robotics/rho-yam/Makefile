# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
.PHONY: setup test smoke
setup:
	uv sync --locked
test:
	uv run --locked pytest -q
smoke:
	uv run --locked python run_policy.py --demo --task 'Pick up the red block and place it in the blue tray.'
