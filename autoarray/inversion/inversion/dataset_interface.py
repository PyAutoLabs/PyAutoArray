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
        sparse_dirty_image=None,
        data_term=None,
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

            For an interferometer inversion on the sparse path this may be `None`, meaning "the raw visibilities
            the `sparse_operator` was built from, unmodified". It is only valid with a `sparse_operator` that
            carries a precomputed `data_term` (built by `Interferometer.apply_sparse_operator` or
            `apply_sparse_operator_from_chunks`): the sparse inversion's data vector already comes from the
            operator's cached dirty image, and `fast_chi_squared` then reads the operator's `data_term` instead
            of reducing over the visibilities, so no visibility array is touched by the likelihood. Output-only
            quantities that need the data itself (e.g. `data_subtracted_dict`) are unavailable in that case.
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
        sparse_dirty_image
            The noise-weighted dirty image `Re(Fᴴ W d)` of this interface's `data`, used by the sparse (w-tilde)
            interferometer inversion to form its data vector. The `sparse_operator` caches the dirty image of the
            visibilities it was built from, so this is only needed when `data` differs from them (e.g. when the
            visibilities of ordinary light profiles have been subtracted). If `None`, the operator's cached dirty
            image is used. This is distinct from `Interferometer.dirty_image`, the unweighted dirty image of the
            data used for visualization.
        data_term
            The chi-squared data term `sum(d_r^2/sigma_r^2) + sum(d_i^2/sigma_i^2)` of *this interface's*
            (possibly profile-subtracted) visibilities, read by the sparse interferometer inversion's
            `fast_chi_squared` when `data` is `None` in preference to the `sparse_operator`'s cached scalar (which
            is the data term of the raw, unsubtracted visibilities). It is how a fit with ordinary light profiles
            on an array-free dataset passes `data=None`: the light profiles' visibilities `F i_p` are never
            formed, and the subtracted data term `data_term - 2 i_p^T d~ + i_p^T W~ i_p` is computed by
            `inversion_interferometer_util.sparse_profile_terms_from` alongside the subtracted
            `sparse_dirty_image`. If `None`, the operator's cached scalar is used. A scalar (traced under
            `jax.jit` when the profile image is).
        """
        self.data = data
        self.noise_map = noise_map
        self.grids = grids
        self.psf = psf
        self.transformer = transformer
        self.sparse_operator = sparse_operator
        self.noise_covariance_matrix = noise_covariance_matrix
        self.sparse_dirty_image = sparse_dirty_image
        self.data_term = data_term

    @property
    def mask(self):
        return self.grids.lp.mask
