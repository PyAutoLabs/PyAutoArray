from autonerves import jax_wrapper
from autonerves.dictable import register_parser
from autonerves import conf

conf.instance.register(__file__)

from . import exc
from . import type
from . import util
from . import fixtures
from . import mock as m

from .dataset import preprocess
from .dataset.abstract.dataset import AbstractDataset
from .dataset.grids import GridsInterface
from .dataset.imaging.dataset import Imaging
from .dataset.imaging.simulator import SimulatorImaging
from .dataset.interferometer.dataset import Interferometer
from .dataset.interferometer.simulator import SimulatorInterferometer
from .dataset.dataset_model import DatasetModel
from .fit.fit_dataset import AbstractFit
from .fit.fit_dataset import FitDataset
from .fit.fit_imaging import FitImaging
from .fit.fit_interferometer import FitInterferometer
from .geometry.geometry_2d import Geometry2D
from .inversion.mesh import mesh
from .inversion.mesh import image_mesh
from .inversion import regularization as reg
from .settings import Settings
from .inversion.inversion.abstract import AbstractInversion
from .preloads import AbstractPreloads
from .preloads import PreloadsImaging
from .preloads import PreloadsInterferometer
from .inversion.regularization.abstract import AbstractRegularization
from .inversion.inversion.factory import inversion_from as Inversion
from .inversion.inversion.dataset_interface import DatasetInterface
from .inversion.mesh.border_relocator import BorderRelocator
from .inversion.pixelization import Pixelization
from .inversion.mappers.abstract import Mapper
from .inversion.mesh.image_mesh.abstract import AbstractImageMesh
from .inversion.mesh.mesh.abstract import AbstractMesh
from .inversion.mesh.interpolator.rectangular import InterpolatorRectangular
from .inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
from .inversion.inversion.imaging.mapping import InversionImagingMapping
from .inversion.inversion.imaging.sparse import InversionImagingSparse
from .inversion.inversion.imaging.inversion_imaging_util import ImagingSparseOperator
from .inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from .inversion.inversion.interferometer.mapping import InversionInterferometerMapping
from .inversion.inversion.interferometer.inversion_interferometer_util import (
    InterferometerSparseOperator,
)
from .inversion.linear_obj.linear_obj import LinearObj
from .inversion.linear_obj.func_list import AbstractLinearObjFuncList
from .mask.derive.indexes_2d import DeriveIndexes2D
from .mask.derive.mask_1d import DeriveMask1D
from .mask.derive.mask_2d import DeriveMask2D
from .mask.derive.grid_1d import DeriveGrid1D
from .mask.derive.grid_2d import DeriveGrid2D
from .mask.derive.zoom_2d import Zoom2D
from .mask.mask_1d import Mask1D
from .mask.mask_2d import Mask2D
from .operators.transformer import TransformerDFT
from .operators.transformer import TransformerNUFFT
from .operators.transformer import TransformerNUFFTPyNUFFT
from .operators.over_sampling.decorator import over_sample
from .operators.contour import Grid2DContour
from .layout.layout import Layout1D
from .layout.layout import Layout2D
from .structures.arrays.uniform_1d import Array1D
from .structures.arrays.uniform_2d import Array2D
from .structures.arrays.rgb import Array2DRGB
from .structures.arrays.irregular import ArrayIrregular
from .structures.grids.uniform_1d import Grid1D
from .structures.grids.uniform_2d import Grid2D
from .operators.over_sampling.over_sampler import OverSampler
from .structures.grids.irregular_2d import Grid2DIrregular
from .inversion.mesh.mesh_geometry.rectangular import MeshGeometryRectangular
from .inversion.mesh.mesh_geometry.delaunay import MeshGeometryDelaunay
from .inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
from .operators.convolver import Convolver
from .operators.interp_2d import interp_2d
from .structures.vectors.uniform import VectorYX2D
from .structures.vectors.irregular import VectorYX2DIrregular
from .structures.triangles.abstract import AbstractTriangles
from .structures.triangles.shape import Circle
from .structures.triangles.shape import Triangle
from .structures.triangles.shape import Square
from .structures.triangles.shape import Polygon
from .structures import decorators as grid_dec  # deprecated alias
from .structures import decorators
from .structures.header import Header
from .layout.region import Region1D
from .layout.region import Region2D
from .structures.visibilities import Visibilities
from .structures.visibilities import VisibilitiesNoiseMap

from autonerves import conf
from autonerves.fitsable import ndarray_via_hdu_from
from autonerves.fitsable import ndarray_via_fits_from
from autonerves.fitsable import header_obj_from
from autonerves.fitsable import output_to_fits
from autonerves.fitsable import hdu_list_for_output_from

conf.instance.register(__file__)

__version__ = "2026.7.23.1"

# ---------------------------------------------------------------------------
# Public re-export of the autonerves configuration / serialization surface.
#
# Workspaces, tutorials and downstream code import these names from the science
# library (e.g. ``from autolens import conf``) rather than depending on the
# ``autonerves`` package directly, so the underlying configuration / serialization
# layer stays an implementation detail of the library.
# ---------------------------------------------------------------------------
from autonerves import conf
from autonerves import jax_wrapper
from autonerves import fitsable
from autonerves import setup_colab
from autonerves import setup_notebook
from autonerves.conf import with_config
from autonerves.dictable import from_dict, from_json, to_dict, output_to_json
from autonerves.fitsable import (
    output_to_fits,
    hdu_list_for_output_from,
    ndarray_via_fits_from,
    ndarray_via_hdu_from,
    header_obj_from,
)
from autonerves.test_mode import (
    with_test_mode_segment,
    skip_visualization,
    skip_fit_output,
    skip_checks,
    is_test_mode,
    test_mode_level,
)
