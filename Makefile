install:
	uv sync --all-groups --all-extras
	uv run pre-commit install

pre-commit:
	uv run pre-commit run --all-files