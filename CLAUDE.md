# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dependency Graph

PyAutoArray depends on **autoconf** (shared configuration and utilities).
PyAutoArray does **NOT** depend on PyAutoFit, PyAutoGalaxy, or PyAutoLens.
Never import from `autofit`, `autogalaxy`, or `autolens` in this repo.
Shared utilities (e.g. `test_mode`, `jax_wrapper`) belong in autoconf.

## Commands

### Install
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
# All tests
python -m pytest test_autoarray/

# Single test file
python -m pytest test_autoarray/structures/test_arrays.py

# With output
python -m pytest test_autoarray/structures/test_arrays.py -s
```

### Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autoarray/
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

### Formatting
```bash
black autoarray/
```

### Plot Output Mode

Set `PYAUTO_OUTPUT_MODE=1` to capture every figure produced by a script into numbered PNG files in `./output_mode/<script_name>/`. This is useful for visually inspecting all plots from an integration test without needing a display.

```bash
PYAUTO_OUTPUT_MODE=1 python scripts/my_script.py
# -> ./output_mode/my_script/0_fit.png, 1_tracer.png, ...
```

When this env var is set, all `save_figure`, `subplot_save`, and `_save_subplot` calls are intercepted — the normal output path is bypassed and figures are written sequentially to the output_mode directory instead.

## Architecture

**PyAutoArray** is the low-level data structures and numerical utilities package for the PyAuto ecosystem. It provides:
- **Grid and array structures** — uniform and irregular 2D grids, arrays, vector fields
- **Masks** — 1D and 2D masks that define which pixels are active
- **Datasets** — imaging and interferometer dataset containers
- **Inversions / pixelizations** — sparse linear algebra for source reconstruction
- **Decorators** — input/output homogenisation for grid-consuming functions

## Core Data Structures

All data structures inherit from `AbstractNDArray` (`abstract_ndarray.py`). Key subclasses:

| Class | Description |
|---|---|
| `Array2D` | Uniform 2D array tied to a `Mask2D` |
| `ArrayIrregular` | Unmasked 1D collection of values |
| `Grid2D` | Uniform (y,x) coordinate grid tied to a `Mask2D` |
| `Grid2DIrregular` | Irregular (y,x) coordinate collection |
| `VectorYX2D` | Uniform 2D vector field |
| `VectorYX2DIrregular` | Irregular vector field |

`AbstractNDArray` provides arithmetic operators (`__add__`, `__sub__`, `__rsub__`, etc.), all decorated with `@to_new_array` and `@unwrap_array` so that operations between autoarray objects and raw scalars/arrays work naturally and return a new autoarray of the same type.

The `.array` property returns the raw underlying `numpy.ndarray` or `jax.Array`:
```python
arr = aa.ArrayIrregular(values=[1.0, 2.0])
arr.array        # raw numpy array
arr._array       # same, internal attribute
```

The constructor unwraps nested autoarray objects automatically:
```python
# while isinstance(array, AbstractNDArray): array = array.array
```

## Decorator System

`autoarray/structures/decorators/` contains three output-wrapping decorators used on all grid-consuming functions. They ensure that the **type of the output structure matches the type of the input grid**:

| Decorator | Grid2D input | Grid2DIrregular input |
|---|---|---|
| `@aa.grid_dec.to_array` | `Array2D` | `ArrayIrregular` |
| `@aa.grid_dec.to_grid` | `Grid2D` | `Grid2DIrregular` |
| `@aa.grid_dec.to_vector_yx` | `VectorYX2D` | `VectorYX2DIrregular` |

### How the decorators work

All three share `AbstractMaker` (`decorators/abstract.py`). The decorator:
1. Wraps the function in a `wrapper(obj, grid, xp=np, *args, **kwargs)` signature
2. Instantiates the relevant `*Maker` class with the function, object, grid, and `xp`
3. `AbstractMaker.result` checks the grid type and calls the appropriate `via_grid_2d` / `via_grid_2d_irr` method to wrap the raw result

The function body receives the grid as-is and **must return a raw array** (not an autoarray wrapper). The decorator does the wrapping:

```python
@aa.grid_dec.to_array
def convergence_2d_from(self, grid, xp=np, **kwargs):
    # grid is Grid2D or Grid2DIrregular — access raw values via grid.array[:,0]
    y = grid.array[:, 0]
    x = grid.array[:, 1]
    return xp.sqrt(y**2 + x**2)   # return raw array; decorator wraps it
```

`AbstractMaker` also stores `use_jax = xp is not np` and exposes `_xp` (either `jnp` or `np`), but the wrapping step always runs regardless of `xp`. Autoarray types are **not registered as JAX pytrees**, so they cannot be directly returned from inside a `jax.jit` trace (see JAX section below).

### Accessing grid coordinates inside a decorated function

Inside a decorated function body, access the raw underlying array with `.array`:

```python
# Correct — works for both numpy and jax backends
y = grid.array[:, 0]
x = grid.array[:, 1]

# Also correct for simple slicing (returns raw array via __getitem__)
y = grid[:, 0]
x = grid[:, 1]
```

The `@transform` decorator (also in `decorators/`) shifts and rotates the input grid to the profile's reference frame before passing it to the function. It calls `obj.transformed_to_reference_frame_grid_from(grid, xp)` (decorated with `@to_grid`) and passes the result as the `grid` argument. After transformation the grid is still an autoarray object; `.array` still works.

### Decorator stacking order

Decorators are applied bottom-up (innermost first). The canonical order for mass/light profile methods is:

```python
@aa.grid_dec.to_array        # outermost: wraps output
@aa.grid_dec.transform       # innermost: transforms grid input
def convergence_2d_from(self, grid, xp=np, **kwargs):
    ...
```

## JAX Support

The `xp` parameter pattern is the single point of control:
- `xp=np` (default) — pure NumPy path
- `xp=jnp` — JAX path; `jax` / `jax.numpy` are only imported locally

### Why autoarray types cannot be returned from `jax.jit`

`AbstractNDArray` subclasses (`Array2D`, `ArrayIrregular`, `VectorYX2DIrregular`, etc.) are **not registered as JAX pytrees**. The `instance_flatten` / `instance_unflatten` class methods are defined on `AbstractNDArray` but are never passed to `jax.tree_util.register_pytree_node`. As a result:

- Constructing an autoarray wrapper **inside** a JIT trace is fine (Python-level code runs normally during tracing)
- **Returning** an autoarray wrapper as the output of a `jax.jit`-compiled function **fails** with `TypeError: ... is not a valid JAX type`

### The `if xp is np:` guard pattern

Functions that are called directly inside `jax.jit` (i.e., as the outermost call in the lambda) must not return autoarray wrappers on the JAX path. The correct pattern is:

```python
def convergence_2d_via_hessian_from(self, grid, xp=np):
    hessian_yy, hessian_xx = ...
    convergence = 0.5 * (hessian_yy + hessian_xx)

    if xp is np:
        return aa.ArrayIrregular(values=convergence)   # numpy: wrapped
    return convergence                                   # jax: raw jax.Array
```

This pattern is used in `autogalaxy/operate/lens_calc.py` for all `LensCalc` methods that are called inside `jax.jit`. It does **not** affect decorated helper functions (like `deflections_yx_2d_from`) because those are called as intermediate steps — their autoarray wrappers are consumed by downstream Python code, never returned as JIT outputs.

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
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
