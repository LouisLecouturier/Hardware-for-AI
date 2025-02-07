# Metrics

## Par batch

-   Temps
-   Loss
-   Consommation electrique
-   Utilisation memoire
-   Utilisation CPU
-   Temperature

## Global

-   Somme temps
-   Best loss
-   Somme consommation electrique
-   Moyenne utilisation memoire
-   Moyenne utilisation CPU
-   Moyenne temperature
-   Temps d'inference (TODO)

## Autre (à voir)

Accuracy/précision
Vitesse de convergence
Stabilité de l'entraînement

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
