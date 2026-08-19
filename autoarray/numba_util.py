import logging
import sys
import threading

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


# Decorated functions are queued here and only handed to numba on the first
# call of any of them, keeping ``import numba`` off the library import path.
# Materialization converts every queued function at once and rebinds the
# defining module's global, because numba's nopython mode must resolve
# cross-calls between decorated functions to real dispatchers at compile time.
_pending = []
_materialize_lock = threading.Lock()


def _materialize_all():
    with _materialize_lock:
        if not _pending:
            return

        try:
            import numba
        except ModuleNotFoundError:
            numba = None

        while _pending:
            func, options, placeholder, state = _pending.pop()

            if numba is None:
                target = func
            else:
                target = numba.jit(func, **options)

            state["target"] = target

            module = sys.modules.get(func.__module__)
            if module is not None and getattr(module, func.__name__, None) is placeholder:
                setattr(module, func.__name__, target)


def jit(nopython=nopython, cache=cache, parallel=parallel, fastmath=False):
    options = dict(
        nopython=nopython,
        cache=cache,
        parallel=parallel,
        fastmath=fastmath,
    )

    def wrapper(func):
        import functools

        state = {"target": None}

        @functools.wraps(func)
        def lazy(*args, **kwargs):
            if state["target"] is None:
                _materialize_all()
            return state["target"](*args, **kwargs)

        _pending.append((func, options, lazy, state))

        return lazy

    return wrapper
