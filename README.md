# Installation

```bash
pip install uv
```

## Install dep

```bash
uv sync
```

## Add libs

```bash
uv add <lib>
```

## Create Jupyter kernel

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```

## Launch file

```bash
uv run <file-name>
```

## Get all commands

```bash
uv
```
