# PyAutoArray — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

## What this repo is

**PyAutoArray** (package `autoarray`) is the low-level data-structure and
numerical-utility layer: masks, arrays, (y,x) grids, imaging/interferometer
datasets, inversions/pixelizations, convolution/over-sampling operators, and
the grid decorators used throughout PyAutoGalaxy and PyAutoLens.

Dependency direction: autoarray depends on **autoconf** only. It does **not**
import `autofit`, `autogalaxy`, or `autolens` — never add such an import.
Shared utilities (e.g. `test_mode`, `jax_wrapper`) belong in autoconf.

## Related repos

- **Source siblings:** PyAutoConf (upstream). PyAutoGalaxy / PyAutoLens build
  directly on autoarray.
- No `_workspace`, `_workspace_test`, or HowTo of its own. The JAX/`xp` path is
  exercised by the parity scripts in **autogalaxy_workspace_test** and
  **autolens_workspace_test**.
- **docs/** — Sphinx source; published to ReadTheDocs.

## Architecture

- `autoarray/structures/` — `Array2D`, `Grid2D`, `Grid2DIrregular`, `VectorYX2D`, and the grid decorators.
- `autoarray/dataset/` — `Imaging` / `Interferometer` dataset containers.
- `autoarray/inversion/` — pixelizations and linear inversion machinery.
- `autoarray/operators/` — `Convolver`, over-sampling, and related operators.
- `test_autoarray/` — test suite.

## Quick commands

```bash
pip install -e ".[dev]"                                  # install with dev/test extras
python -m pytest test_autoarray/                         # full test suite
python -m pytest test_autoarray/structures/test_arrays.py   # one focused test (add -s for output)
black autoarray/                                         # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autoarray/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Configuration & defaults

autoconf supplies the packaged defaults under `autoarray/config/`. Workspaces
override them via their own `config/` directory; the test suite pushes a local
config dir via `conf.instance.push(...)` in `test_autoarray/conftest.py`. When
a change adds a new config key, mirror it into the packaged defaults so
downstream workspaces inherit it.

## JAX & `xp`

NumPy is the default everywhere; JAX is opt-in and never imported at module
level. The `xp` parameter is the single point of control:

- `xp=np` (default) — pure NumPy path.
- `xp=jnp` — JAX path; `jax` / `jax.numpy` imported locally inside the function.

Thread `xp` through **every** nested call (`self.X()`, helpers, properties) — a
missed site silently defaults to `xp=np` and fails when a tracer hits an `np.*`
op. Two patterns cross the `jax.jit` boundary: the `if xp is np:` **guard** for
functions that return a raw `jax.Array`, and **pytree registration**
(`abstract_ndarray._register_as_pytree` / `register_instance_pytree`) for
functions that return a real wrapper or structured object.

**Unit tests are NumPy-only.** A JAX/`xp` change is validated only by the
parity scripts in `autogalaxy_workspace_test` / `autolens_workspace_test`
(`jax.jit` round-trip + `fitness._vmap` batch eval), not by `test_autoarray/`.

Full detail (decorator internals, `xp`-threading hazards, both JIT patterns):
**[`docs/agents/jax_and_decorators.md`](docs/agents/jax_and_decorators.md)**.

## Public API

The public surface is defined authoritatively in `autoarray/__init__.py` — read
it rather than trusting a hand-maintained list. Canonical import:

```python
import autoarray as aa
```

Core types (`Array2D`, `Grid2D`, `Grid2DIrregular`, `VectorYX2D`, …) inherit
from `AbstractNDArray`; `.array` returns the raw `numpy.ndarray` / `jax.Array`.

## Key rules / footguns

- Import direction: autoconf only — never `autofit` / `autogalaxy` / `autolens`.
- Grid-consuming functions decorated with `@aa.decorators.to_array` / `to_grid`
  / `to_vector_yx` must return a **raw array** — the decorator wraps it. (Write
  `aa.decorators.*`; `aa.grid_dec` is a deprecated alias.)
- Access grid coordinates via `grid.array[:, 0]`, not `grid[:, 0]`.
- All files use Unix line endings (LF, `\n`) — never `\r\n`.

## Working on issues

1. Read the issue description and any linked plan.
2. Identify affected files and make the change.
3. Run the full suite: `python -m pytest test_autoarray/`.
4. If you changed public API, say so explicitly — downstream packages
   (PyAutoGalaxy, PyAutoLens) and the workspaces may need updates.
5. Ensure all tests pass before opening a PR.

## Deep dives

- [`docs/agents/jax_and_decorators.md`](docs/agents/jax_and_decorators.md) —
  decorator system, `xp` backend pattern, and the `jax.jit` boundary.

<!-- repos_sync:history:begin -->
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
<!-- repos_sync:history:end -->
