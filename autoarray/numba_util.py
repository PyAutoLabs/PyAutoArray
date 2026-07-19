import logging

from autonerves import conf


logger = logging.getLogger(__name__)

try:
    nopython = conf.instance["general"]["numba"]["nopython"]
    cache = conf.instance["general"]["numba"]["cache"]
    parallel = conf.instance["general"]["numba"]["parallel"]
except Exception:
    nopython = True
    cache = True
    parallel = False


def jit(nopython=nopython, cache=cache, parallel=parallel, fastmath=False):

    def wrapper(func):

        try:

            import numba

            return numba.jit(
                func,
                nopython=nopython,
                cache=cache,
                parallel=parallel,
                fastmath=fastmath,
            )

        except ModuleNotFoundError:

            return func

    return wrapper
