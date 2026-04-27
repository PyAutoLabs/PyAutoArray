# PyAutoArray — Agent Instructions

**PyAutoArray** is the low-level data structures and numerical utilities package for the PyAuto ecosystem. It provides grids, masks, arrays, datasets, inversions, and the decorator system used throughout PyAutoGalaxy and PyAutoLens.

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest test_autoarray/
python -m pytest test_autoarray/structures/test_arrays.py
python -m pytest test_autoarray/structures/test_arrays.py -s
```

### Sandboxed / Codex runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autoarray/
```

## Key Architecture

- **Data structures**: `Array2D`, `Grid2D`, `Grid2DIrregular`, `VectorYX2D` — all inherit from `AbstractNDArray`
- **Decorator system** (`structures/decorators/`): `@to_array`, `@to_grid`, `@to_vector_yx`, `@transform` — ensures output type matches input grid type
- **Datasets**: `Imaging`, `Interferometer` — containers for observational data
- **Inversions** (`inversion/`): sparse linear algebra for source reconstruction via pixelizations
- **Operators**: `Convolver` (PSF convolution), over-sampling utilities

## Key Rules

- The `xp` parameter pattern controls NumPy vs JAX: `xp=np` (default) or `xp=jnp`
- Autoarray types are **not** JAX pytrees — they cannot be returned from `jax.jit` functions
- Decorated functions must return **raw arrays**, not autoarray wrappers
- All files must use Unix line endings (LF)
- Format with `black autoarray/`

## Working on Issues

1. Read the issue description and any linked plan.
2. Identify affected files and write your changes.
3. Run the full test suite: `python -m pytest test_autoarray/`
4. Ensure all tests pass before opening a PR.
5. If changing public API, note the change in your PR description — downstream packages (PyAutoGalaxy, PyAutoLens) and workspaces may need updates.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
