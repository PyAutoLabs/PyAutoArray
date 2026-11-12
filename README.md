# PyAutoArray: Arrays, Grids and Masks for Astronomy

**PyAutoArray** provides arrays, masks, coordinate grids and numerical tools for astronomical imaging and interferometer data. It powers the data handling and image reconstruction in **PyAutoGalaxy** and **PyAutoLens**.

If you want to model galaxies or gravitational lenses, start with [PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy) or [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens). Use PyAutoArray directly when you need these building blocks in your own Python code.

## Getting Started

Install the package:

```bash
pip install autoarray
```

Create a two-dimensional array with a pixel scale:

```python
import autoarray as aa

image = aa.Array2D.no_mask(
    values=[[1.0, 2.0], [3.0, 4.0]],
    pixel_scales=0.1,
)
```

Create a circular mask and a grid of coordinates for its unmasked pixels:

```python
mask = aa.Mask2D.circular(
    shape_native=(50, 50),
    pixel_scales=0.1,
    radius=2.0,
)

grid = aa.Grid2D.from_mask(mask=mask)
```

## Explore Further

- **[Galaxy modeling examples](https://github.com/PyAutoLabs/autogalaxy_workspace#find-your-examples)** — see arrays, grids and masks used in galaxy analysis.
- **[Lensing examples](https://github.com/PyAutoLabs/autolens_workspace#find-your-examples)** — explore their use in gravitational lens modeling.
- **[Source code](https://github.com/PyAutoLabs/PyAutoArray/tree/main/autoarray)** — explore the data structures and numerical tools.
- **[Tests](https://github.com/PyAutoLabs/PyAutoArray/tree/main/test_autoarray)** — find focused examples of individual components.

## Community and Contributing

Ask questions, suggest improvements and share your work in [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions), or chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

To contribute, see the [contribution guide](https://github.com/PyAutoLabs/PyAutoArray/blob/main/CONTRIBUTING.md).

## License

See the [license](https://github.com/PyAutoLabs/PyAutoArray/blob/main/LICENSE).

<sub><i><a href="https://fredagain.lnk.to/places-to-be">me next to you / and you next to me</a></i></sub>
