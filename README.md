# PyAutoArray

**PyAutoArray** (package `autoarray`) is the low-level data-structure and
numerical-utility layer of the [PyAuto](https://github.com/PyAutoLabs)
ecosystem. It provides masks, arrays, (y,x) coordinate grids,
imaging/interferometer datasets, inversions/pixelizations for source
reconstruction, and convolution/over-sampling operators.

`PyAutoGalaxy` and `PyAutoLens` build directly on autoarray: every grid a
profile consumes, every masked image a fit operates on, and the linear-algebra
inversions behind pixelized source reconstruction are autoarray objects. The
package supports both a NumPy and an opt-in JAX (`xp=jnp`) backend.

## Install

```bash
pip install autoarray
```

## Examples

A masked 2D array tied to a pixel scale:

```python
import autoarray as aa

arr = aa.Array2D.no_mask(values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.1)
arr.shape_native        # (2, 2)
arr.native[0, 0]        # 1.0
```

A circular mask and the (y,x) coordinate grid of its unmasked pixels:

```python
mask = aa.Mask2D.circular(shape_native=(50, 50), pixel_scales=0.1, radius=2.0)
mask.pixels_in_mask     # 1264

grid = aa.Grid2D.from_mask(mask=mask)       # shape (1264, 2)
uniform = aa.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)
```

A normalized Gaussian PSF convolver:

```python
convolver = aa.Convolver.from_gaussian(
    shape_native=(11, 11), pixel_scales=0.1, sigma=1.0, normalize=True
)
convolver.kernel.shape_native       # (11, 11)
convolver.kernel.array.sum()        # 1.0
```

## Links

- Source & tests: [`autoarray/`](autoarray), [`test_autoarray/`](test_autoarray)
- Decorators & JAX deep dive: [`docs/agents/jax_and_decorators.md`](docs/agents/jax_and_decorators.md)
- Agent/contributor instructions: [`AGENTS.md`](AGENTS.md)
- Ecosystem: [PyAutoLabs on GitHub](https://github.com/PyAutoLabs)
