class DatasetInterface:
    def __init__(
        self,
        data,
        noise_map,
        grids=None,
        psf=None,
        transformer=None,
        sparse_operator=None,
        noise_covariance_matrix=None,
    ):
        """
        Generic class which acts as an interface between a dataset and an inversion.

        The inputs to the inversion module are an `Imaging` or `Interferometer` dataset, and it is recommend
        instances of these objects should be input into an inversion whenever possible.

        However, the dataset's attributes are commonly modified before being input into the inversion, for example:

        - In PyAutoGalaxy and PyAutoLens, the data may have the light profiles of certain galaxies or the background
        sky subtracted from it.

        - The noise-map may be scaled to put large values in regions of the data identified to be difficult to fit.

        - The PSF may be a part of the model where it is customized by free parameters which vary.

        In all cases, the dataset's attributes (modified and unmodified) can be passed through this class and into the
        inversion.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        grids
            The grids of (y,x) Cartesian coordinates that the image data is paired with, which are used for evaluating
            light profiles and calculations associated with a pixelization.
        psf
            Perform 2D convolution of the imaging data's PSF when computing the operated mapping matrix.
        transformer
            Performs a Fourier transform of the image-data from real-space to visibilities when computing the
            operated mapping matrix.
        sparse_operator
            The sparse_operator matrix used by the w-tilde formalism to construct the data vector and
            curvature matrix during an inversion efficiently..
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        """
        self.data = data
        self.noise_map = noise_map
        self.grids = grids
        self.psf = psf
        self.transformer = transformer
        self.sparse_operator = sparse_operator
        self.noise_covariance_matrix = noise_covariance_matrix

    @property
    def mask(self):
        return self.grids.lp.mask
