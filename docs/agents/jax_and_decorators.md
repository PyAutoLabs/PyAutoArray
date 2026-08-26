# JAX & the decorator system — deep dive

Long-form reference for the grid decorators, the `xp` (NumPy/JAX) backend
pattern, and how autoarray types cross the `jax.jit` boundary. The per-repo
`AGENTS.md` files keep only a short summary and link here. This is the single
canonical source for the detail — PyAutoGalaxy and PyAutoLens point at it
rather than re-explaining.

Everything below is grounded in the installed source under
`autoarray/`, `autogalaxy/`, and `autolens/`. Where a class or function is
named, it exists in the current tree.

---

## 1. The decorator system

`autoarray/structures/decorators/` contains the output-wrapping decorators
used on all grid-consuming functions. They ensure the **type of the output
structure matches the type of the input grid**.

Import them as `aa.decorators.*`. (`aa.grid_dec` still resolves as a
**deprecated alias** — `autoarray/__init__.py` defines
`from .structures import decorators as grid_dec  # deprecated alias` — but
every shipped profile uses `aa.decorators.*`, so write that form.)

| Decorator | `Grid2D` input → | `Grid2DIrregular` input → |
|---|---|---|
| `@aa.decorators.to_array` | `Array2D` | `ArrayIrregular` |
| `@aa.decorators.to_grid` | `Grid2D` | `Grid2DIrregular` |
| `@aa.decorators.to_vector_yx` | `VectorYX2D` | `VectorYX2DIrregular` |

### How they work

All three share `AbstractMaker` (`decorators/abstract.py`). The decorator:

1. Wraps the function in a `wrapper(obj, grid, xp=np, *args, **kwargs)` signature.
2. Instantiates the relevant `*Maker` class with the function, object, grid, and `xp`.
3. `AbstractMaker.result` checks the grid type and calls the appropriate
   `via_grid_2d` / `via_grid_2d_irr` method to wrap the raw result.

The function body receives the grid as-is and **must return a raw array**
(not an autoarray wrapper). The decorator does the wrapping:

```python
@aa.decorators.to_array
def convergence_2d_from(self, grid, xp=np, **kwargs):
    # grid is Grid2D or Grid2DIrregular — access raw values via grid.array[:, 0]
    y = grid.array[:, 0]
    x = grid.array[:, 1]
    return xp.sqrt(y**2 + x**2)   # return raw array; decorator wraps it
```

`AbstractMaker` stores `use_jax = xp is not np` and exposes `_xp` (either `jnp`
or `np`), but the wrapping step always runs regardless of `xp`.

### Accessing grid coordinates inside a decorated function

Access the raw underlying array with `.array`:

```python
# Correct — works for both numpy and jax backends
y = grid.array[:, 0]
x = grid.array[:, 1]

# Also works for simple slicing (returns raw array via __getitem__)
y = grid[:, 0]
x = grid[:, 1]
```

Prefer `grid.array[:, 0]` — after `@transform` the grid is still an autoarray
object and `.array` is the safe way to extract the underlying data for both
numpy and jax backends.

### `@transform` and stacking order

`@aa.decorators.transform` shifts and rotates the input grid to the profile's
reference frame before passing it to the function. It calls
`obj.transformed_to_reference_frame_grid_from(grid, xp)` (itself decorated with
`@to_grid`) and passes the result as the `grid` argument. After transformation
the grid is still an autoarray object; `.array` still works. Some call sites
pass `rotate_back=True` (e.g. `@aa.decorators.transform(rotate_back=True)`).

Decorators apply bottom-up (innermost first). The canonical order for
mass/light profile methods is:

```python
@aa.decorators.to_array      # outermost: wraps output
@aa.decorators.transform     # innermost: transforms grid input
def convergence_2d_from(self, grid, xp=np, **kwargs):
    ...
```

---

## 2. `AbstractNDArray` and the `.array` property

All data structures inherit from `AbstractNDArray` (`abstract_ndarray.py`).
Key subclasses: `Array2D`, `ArrayIrregular`, `Grid2D`, `Grid2DIrregular`,
`VectorYX2D`, `VectorYX2DIrregular`.

`AbstractNDArray` provides arithmetic operators (`__add__`, `__sub__`,
`__rsub__`, …) so operations between autoarray objects and raw scalars/arrays
return a new autoarray of the same type. The `.array` property returns the raw
underlying `numpy.ndarray` or `jax.Array`:

```python
arr = aa.ArrayIrregular(values=[1.0, 2.0])
arr.array        # raw numpy (or jax) array
arr._array       # same, internal attribute
```

The constructor unwraps nested autoarray objects automatically
(`while isinstance(array, AbstractNDArray): array = array.array`).

---

## 3. The `xp` backend pattern

The codebase is designed so that **NumPy is the default everywhere and JAX is
opt-in**. JAX is never imported at module level — only locally inside functions
when explicitly requested. The `xp` parameter is the single point of control:

- `xp=np` (default throughout) — pure NumPy path, no JAX dependency at runtime.
- `xp=jnp` — JAX path; `jax` / `jax.numpy` imported locally inside the function.

When adding a new function that should support JAX:

1. Default the parameter to `xp=np`.
2. Guard any JAX imports with `if xp is not np:` and import `jax` / `jax.numpy`
   locally inside that branch.
3. Add the NumPy implementation as the default path.
4. Add a JAX implementation in the guarded branch (e.g. `jax.jacfwd`,
   `jnp.vectorize`).

### Threading `xp` through nested calls

Adding `xp=np` to a method body and swapping `np.*` for `xp.*` is **only half
the work**. Every nested call inside that body — `self.X()`, `obj.X()`, a helper
in `convert.py`, an inherited `@property`, or a sibling method — must also
receive `xp=xp` if it can route to numpy operations on what would otherwise be
JAX tracers. Otherwise the inner call silently defaults to `xp=np` and fails
when a tracer reaches an `np.*` op.

Concrete hazards seen in this codebase:

- **`@property` chains that hardcode `np`.** A property takes no kwargs, so an
  xp-aware caller must either inline the computation under `if xp is not np:`
  or convert the property to a method. Read every `@property` you call from
  xp-aware code; if it does `np.sqrt(...)`, it is a hazard.
- **Inherited methods.** A method may accept `xp` but a call site forgets to
  pass it. Within xp-aware functions, grep for `self.X(` / `obj.X(` and verify
  `xp=xp` is threaded.
- **`convert.py` helpers.** Helpers like `axis_ratio_and_angle_from`,
  `angle_from`, `multipole_comps_from` all take `xp=np`; call sites must thread
  it. They also use Python `&` on JAX bool tracers, which silently calls
  `__array__()` — replace with `xp.logical_and`.
- **`@cached_property` on traced arrays.** Caches a tracer in `self.__dict__`,
  which is invalid across `vmap` batch elements (different batches share the
  cache). Use plain `@property` for any value that depends on JAX-traced inputs.

---

## 4. Crossing the `jax.jit` boundary — two patterns

Autoarray types **are** registered as JAX pytrees (see
`abstract_ndarray._register_as_pytree` and `register_instance_pytree`), so a
wrapper *can* be returned from a jitted function once its class is registered.
Two patterns coexist depending on what the function returns:

### Pattern 1 — `if xp is np:` guard (raw `jax.Array` return)

Functions intended to be called directly inside `jax.jit` as the outermost op,
where no wrapper is needed on the JAX path, guard their autoarray wrapping:

```python
def convergence_2d_via_hessian_from(self, grid, xp=np):
    convergence = 0.5 * (hessian_yy + hessian_xx)

    if xp is np:
        return aa.ArrayIrregular(values=convergence)  # numpy: wrapped
    return convergence                                  # jax: raw jax.Array
```

All `LensCalc` hessian-derived methods (`convergence_2d_via_hessian_from`,
`shear_yx_2d_via_hessian_from`, `magnification_2d_via_hessian_from`,
`magnification_2d_from`, `tangential_eigen_value_from`,
`radial_eigen_value_from`) use this pattern in
`autogalaxy/operate/lens_calc.py` and return raw `jax.Array` on the JAX path.
Intermediate helpers (e.g. `deflections_yx_2d_from`) do **not** need the guard
— their autoarray wrappers are consumed by downstream Python before any JIT
boundary.

### Pattern 2 — pytree-registered wrapper return

Functions that must return a real autoarray wrapper (or a structured object
built from them) rely on JAX pytree registration:

- `AbstractNDArray` auto-registers its subclass with `jax.tree_util` the first
  time an instance is built with `xp=jnp`, via
  `autoarray.abstract_ndarray._register_as_pytree`.
- Higher-level types (`FitImaging`, `Tracer`, `DatasetModel`) use
  `autoarray.abstract_ndarray.register_instance_pytree(cls, no_flatten=...)`,
  which flattens `__dict__` and carries `no_flatten` names through `aux_data`
  for per-analysis constants (dataset, settings, cosmology).
- `AnalysisImaging._register_fit_imaging_pytrees` wires these up when
  `use_jax=True`, so `jax.jit(analysis.fit_from)(instance)` returns a real
  `FitImaging` with `jax.Array` leaves.

---

## 5. Validation — unit tests are NumPy-only

Library unit tests (`test_autoarray/`, `test_autogalaxy/`, `test_autolens/`)
always run on the NumPy path. **No `xp=jnp` JAX assertion belongs in a library
unit test.** A JAX / `xp` change is validated only by the parity scripts in the
`*_workspace_test` repos.

**`jax.jit(fn)(concrete_instance)` is NOT a sufficient JAX trace check.** A
`ModelInstance` with concrete float parameters propagates as floats through
`np.*` ops without raising — an un-threaded `xp` bug stays hidden. Use
`jax.vmap(fitness)(jnp.array(params))` (or `Fitness._vmap` on autofit's
wrapper) instead: vmap forces tracer propagation through every leaf and exposes
un-threaded `xp` sites.

When adding a JAX path to an Analysis class, the workspace_test parity script
must include **both** a `jax.jit(analysis.fit_from)(instance)` round-trip
**and** a `fitness._vmap(parameters)` batch evaluation. PyAutoArray has no own
`autoarray_workspace_test`; array-level JAX changes are exercised downstream in
`autogalaxy_workspace_test/scripts/jax_likelihood_functions/` and the
`autolens_workspace_test` equivalents.

---

## 6. Registering a new type — classification heuristic

When a jit trace fails on an unregistered type (a `jax.tree_util`
`TypeError: unhashable type`, a `NotImplementedError`, or a tracer error whose
frame above names the class), read the offending class's `__init__` /
`__dict__` and classify its attributes before registering:

- **All attrs are `jax.Array` / `np.ndarray` / `AbstractNDArray` / primitives**
  → register with `no_flatten=()` (everything dynamic).
- **Known-aux patterns** — `cosmology`, `settings`, `config`, `dataset`, `psf`,
  `mask`, `_cache`, `_compiled`, any `scipy.spatial.*`, `Transformer*`,
  `PointSolver*` → register with those names in `no_flatten`; they are
  per-analysis constants, not traced leaves.
- **A callable attribute** → do not guess. Callable state is the case that
  silently breaks gradients; decide aux / dynamic / split deliberately.

Register at the type's wiring site (e.g. `_register_fit_imaging_pytrees` in
PyAutoLens) with a one-line comment naming the variant that needed it, then
re-trace. Registration is iterative — each pass usually surfaces the next
unregistered type one frame further in — but if a handful of passes produce no
progress, the type is telling you it holds state that cannot be flattened;
stop and treat it as a design question rather than adding more registrations.

The parity scripts under
`autolens_workspace_test/scripts/jax_likelihood_functions/` are where this is
exercised; `<variant>_pytree.py` asserts the round trip against the NumPy
reference:

```python
ref = analysis.fit_from(instance).log_likelihood                              # NumPy reference
jit_ll = jax.jit(lambda i: analysis.fit_from(i).log_likelihood)(instance)     # jit round-trip
assert jnp.allclose(ref, jit_ll, rtol=1e-4), f"divergence: {ref} vs {jit_ll}"
```

That round trip proves the types flatten; it does **not** prove `xp` is threaded
— pair it with the `fitness._vmap(parameters)` check in §5.
