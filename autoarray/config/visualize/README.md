The `config` folder contains configuration files which customize default **PyAutoLens**.

# Files

- `general.yaml`: Customizes general visualization settings (e.g. the matplotlib backend).
- `include.yaml`: Customize features that appears on plotted images by default (e.g. a mask, a grid).
- `plots.yaml`: Customize which figures are output during a model-fit.
- `mat_wrap.yaml`: Specify the default matplotlib settings when figures and subplots are plotted.
- `mat_wrap_1d.yaml`: Specify the default matplotlib settings when 1D figures and subplots are plotted.
- `mat_wrap_2d.yaml`: Specify the default matplotlib settings when 2D figures and subplots are plotted.

# Changing the colormap

Every 2D figure — imaging data, fits, residual maps, inversion reconstructions —
draws with the colormap named by the `colormap` key of `general.yaml`:

```yaml
colormap: autoarray   # any matplotlib colormap name, e.g. magma, viridis, inferno
```

`autoarray` is the colormap bundled with **PyAutoArray**; any other value is
looked up in matplotlib, so `magma`, `viridis`, `inferno`, `plasma`, `jet` and
the rest of `list(matplotlib.colormaps)` all work. Editing this one key is
enough — no plotting code needs changing.

A name matplotlib does not recognise (a typo, say) raises a `ValueError` naming
the key and the offending value. It is **not** silently swapped back for the
default, so a colormap setting never goes quietly ignored.

## One figure at a time

To override the colormap for a single figure without touching config, pass
`colormap=` to any plot function:

```python
import autoarray.plot as aplt

aplt.plot_array(array=image, colormap="magma")
aplt.plot_inversion_reconstruction(pixel_values=values, mapper=mapper, colormap="viridis")
```

The same argument exists on the **PyAutoGalaxy** and **PyAutoLens** plot
functions (`subplot_fit`, `plot_tracer`, `subplot_sensitivity`, …), which pass
it straight through to **PyAutoArray**. Its "use the config value" default is
spelled `None` in **PyAutoArray** and **PyAutoLens**, and `"default"` in
**PyAutoGalaxy**; both mean the same thing.

## Figures that deliberately ignore the setting

A few figures fix their colormap because the colormap carries meaning that a
user preference should not override:

- The `array_overlay` of `plot_array` uses `Greys`, so the overlaid array stays
  legible on top of whatever colormap the main array is drawn in.
- The weak-lensing figures in **PyAutoLens** use `twilight` for position angles
  (cyclic data needs a cyclic colormap) and `RdBu_r` for residuals (diverging
  data needs a diverging colormap centred on zero).
- The cluster figures in **PyAutoLens** use `gnuplot2`, and the interactive GUI
  tools use `jet`, to keep faint features visible while masks are drawn by hand.
