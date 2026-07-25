import logging
import os
from pathlib import Path
from typing import Union, List, Optional

from autoarray.structures.abstract_structure import Structure

logger = logging.getLogger(__name__)


class Output:
    def __init__(
        self,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        format: Union[str, List[str]] = None,
        format_folder: bool = False,
        bypass: bool = False,
        bbox_inches: str = "tight",
        **kwargs,
    ):
        """
        Sets how the figure or subplot is output, either by displaying it on the screen or writing it to hard-disk.

        This object wraps the following Matplotlib methods:

        - plt.show: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html
        - plt.savefig: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disk
        as a file.

        Parameters
        ----------
        path
            If the figure is output to hard-disk the path of the folder it is saved to.
        filename
            If the figure is output to hard-disk the filename used to save it.
        prefix
            A prefix appended before the file name, e.g. ("prefix_filename").
        prefix
            A prefix appended after the file name, e.g. ("filenam_suffix").
        format
            The format of the output, 'show' displays on the computer screen, 'png' outputs to .png, 'fits' outputs to
            `.fits` format.
        format_folder
            If `True`, all images are output in a folder giving the format name, for
            example `path/to/output/png/filename.png`. This can make managing large image catalogues easier.
        bypass
            Whether to bypass the `plt.show` or `plt.savefig` methods, used when plotting a subplot.
        """
        self.path = path

        self.filename = filename
        self.prefix = prefix
        self.suffix = suffix
        self._format = format
        self.format_folder = format_folder
        self.bypass = bypass
        self.bbox_inches = bbox_inches
        self._tag_fits_multi = None

        self.kwargs = kwargs

    @property
    def format(self) -> str:
        """The output format string; defaults to ``"show"`` when none was given."""
        if self._format is None:
            return "show"
        return self._format

    @property
    def format_list(self):
        """The output format(s) as a list, so iteration always works."""
        if not isinstance(self.format, list):
            return [self.format]
        return self.format

    def output_path_from(self, format):
        """Return the directory path for *format*, creating it if necessary.

        When *format* is ``"show"`` returns ``None`` (no file is written).
        When ``format_folder`` is ``True`` the format name is appended as a
        sub-directory so that ``png`` and ``pdf`` outputs are kept separate.

        Parameters
        ----------
        format
            File format string, e.g. ``"png"``, ``"pdf"``, or ``"show"``.

        Returns
        -------
        str or None
            Absolute path to the output directory, or ``None`` for
            ``format == "show"``.
        """
        if format == "show":
            return None

        if self.format_folder:
            output_path = str(Path(self.path) / format)
        else:
            output_path = self.path

        os.makedirs(output_path, exist_ok=True)

        return output_path

    def filename_from(self, auto_filename):
        """Build the final filename string by applying prefix / suffix.

        When no explicit ``filename`` was passed to ``__init__`` the
        *auto_filename* supplied by the calling plotter is used as the base.

        Parameters
        ----------
        auto_filename
            Fallback filename (without extension) when ``self.filename`` is
            ``None``.

        Returns
        -------
        str
            The resolved filename with any configured prefix and suffix
            applied.
        """
        filename = auto_filename if self.filename is None else self.filename

        if self.prefix is not None:
            filename = f"{self.prefix}{filename}"

        if self.suffix is not None:
            filename = f"{filename}{self.suffix}"

        return filename

    def savefig(self, filename: str, output_path: str, format: str):
        """Call ``plt.savefig`` with the configured ``bbox_inches`` setting.

        Catches ``ValueError`` exceptions (e.g. unsupported format) and logs
        them without raising, so a single bad output format does not abort
        the whole script.

        Parameters
        ----------
        filename
            Base file name without extension.
        output_path
            Directory to write the file (must already exist).
        format
            File format extension string, e.g. ``"png"``.
        """
        import matplotlib.pyplot as plt

        try:
            plt.savefig(
                str(Path(output_path) / f"{filename}.{format}"),
                bbox_inches=self.bbox_inches,
                pad_inches=0.1,
            )
        except ValueError as e:
            logger.info(
                f"""
                Failed to output figure as a .{format} or .fits due to the following error:

                {e}
            """
            )

    def to_figure(
        self, structure: Optional[Structure], auto_filename: Optional[str] = None
    ):
        """
        Output the figure, by either displaying it on the user's screen or to the hard-disk as a .png or .fits file.

        Parameters
        ----------
        structure
            The 2D array of image to be output, required for outputting the image as a fits file.
        auto_filename
            If the filename is not manually specified this name is used instead, which is defined in the parent plotter.
        """
        import matplotlib.pyplot as plt

        filename = self.filename_from(auto_filename=auto_filename)

        for format in self.format_list:
            output_path = self.output_path_from(format=format)

            if format != "show":
                os.makedirs(output_path, exist_ok=True)

            if not self.bypass:
                if os.environ.get("PYAUTO_OUTPUT_MODE") == "1":
                    return self.to_figure_output_mode(filename=filename)

                if format == "show":
                    plt.show()
                elif format == "png" or format == "pdf":
                    self.savefig(filename, output_path, format)

    def subplot_to_figure(
        self, auto_filename: Optional[str] = None, also_show: bool = False
    ):
        """
        Output a subplot figure, either as an image on the screen or to the hard-disk as a png or fits file.

        Parameters
        ----------
        auto_filename
            If the filename is not manually specified this name is used instead, which is defined in the parent plotter.
        """
        import matplotlib.pyplot as plt

        filename = self.filename_from(auto_filename=auto_filename)

        for format in self.format_list:
            output_path = self.output_path_from(format=format)

            if format != "show":
                os.makedirs(output_path, exist_ok=True)

            if os.environ.get("PYAUTO_OUTPUT_MODE") == "1":
                return self.to_figure_output_mode(filename=filename)

            if format == "show":
                plt.show()
            elif format == "png" or format == "pdf":
                self.savefig(filename, output_path, format)

        if also_show:
            plt.show()

    def to_figure_output_mode(self, filename: str):
        """Save the current figure as a numbered PNG snapshot in *output mode*.

        Output mode is activated by setting the environment variable
        ``PYAUTO_OUTPUT_MODE=1``.  Each call increments a global counter
        so that figures are saved as ``0_filename.png``, ``1_filename.png``,
        etc. in a sub-directory named after the running script.  This is useful
        for collecting a sequence of figures during automated testing or
        demonstration scripts.

        Parameters
        ----------
        filename
            Base file name (without extension) for this figure.
        """
        global COUNT

        try:
            COUNT += 1
        except NameError:
            COUNT = 0

        import sys

        script_name = Path(sys.argv[0]).name.replace(".py", "")

        output_path = str(Path(os.getcwd()) / "output_mode" / script_name)
        os.makedirs(output_path, exist_ok=True)

        self.savefig(f"{COUNT}_{filename}", output_path, "png")
