"""
The compute setup of a likelihood: which backend, device, operator, precision and
(for interferometry) transformer evaluate it, and where that choice came from.

:class:`Setup` is the setup tuple of the profiling campaigns, written as a setup_id
``<backend>_<device>_<operator>_<precision>[_<transformer>]`` (for example
``numba_cpu_sparse_fp64``, ``jax_gpu_dense_fp64`` or ``jax_gpu_sparse_fp64_wtilde``),
plus its provenance.

:meth:`Setup.auto` chooses a setup from the rule table ``autoarray/config/setup_rules.yaml``,
which is generated from the PyAutoPulse designation decisions by
:mod:`autoarray.setup.generate_rules`. Detecting the hardware and parsing the
``PYAUTO_SETUP`` environment variable belong to :mod:`autonerves.environment`; this
module only chooses.
"""

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from autonerves.environment import (
    SETUP_BACKENDS,
    SETUP_DEVICES,
    SETUP_ENV_VAR,
    SETUP_PRECISIONS,
    Environment,
    parse_setup_intent,
)

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "setup_rules.yaml"

HARDWARE_CLASS_CPU = "cpu_many_core"
HARDWARE_CLASS_GPU = "gpu_a100"

# The fewest CPU cores a machine may have and still count as the ``cpu_many_core``
# hardware class the Pulse designations were measured on. A CPU-only machine with
# fewer cores (a laptop) uses the ``cpu_many_core`` rule with ``extrapolated=True``.
MANY_CORE_MIN_CORES = 16

# The GPU name substring of the ``gpu_a100`` hardware class. Any other GPU uses the
# ``gpu_a100`` rule with ``extrapolated=True``.
GPU_A100_NAME = "A100"

EXPLICIT_RULE_ID = f"explicit:{SETUP_ENV_VAR}"


def _validate_token(name: str, value: Optional[str], allowed=None, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, str):
        raise ValueError(f"Setup {name} must be a string, got {value!r}.")
    if allowed is not None:
        if value not in allowed:
            raise ValueError(
                f"Setup {name}={value!r} is not valid; allowed: {allowed}."
            )
        return
    if not (value and value.isascii() and value.isalnum() and value == value.lower()):
        raise ValueError(
            f"Setup {name}={value!r} is not valid; it must be a non-empty lowercase "
            "alphanumeric token (e.g. 'dense', 'sparse', 'na', 'wtilde', 'nufft')."
        )


@functools.lru_cache(maxsize=None)
def _load_rules(path: str = str(RULES_PATH)) -> Tuple[dict, ...]:
    """
    The rows of the setup rule table, loaded once and cached.

    Parameters
    ----------
    path
        The rule table to load; defaults to the table shipped with autoarray.
    """
    with open(path) as f:
        rules = yaml.safe_load(f)["rules"]
    return tuple(rules)


def rules() -> List[dict]:
    """
    A copy of the rows of the setup rule table ``autoarray/config/setup_rules.yaml``.

    Each row has the keys ``likelihood`` (``<dataset_type>/<variant>``), ``regime``,
    ``hardware_class``, ``optimal`` (a setup_id), ``secondary`` (a list of setup_ids),
    ``decision_id`` (the PyAutoPulse decision id) and ``rule`` (one sentence).
    """
    return [dict(row) for row in _load_rules()]


def _rule_index() -> Dict[Tuple[str, str, str], dict]:
    return {
        (row["likelihood"], row["regime"], row["hardware_class"]): row
        for row in _load_rules()
    }


@dataclass(frozen=True)
class Setup:
    """
    A compute setup for evaluating a likelihood, plus where it came from.

    The first five fields are the setup tuple, written as the setup_id
    ``<backend>_<device>_<operator>_<precision>[_<transformer>]`` (see
    :attr:`setup_id`). The remaining fields record how the setup was chosen and do
    not take part in the setup_id.

    Parameters
    ----------
    backend
        The array backend: ``"jax"``, ``"numba"`` or ``"numpy"``.
    device
        ``"cpu"`` or ``"gpu"``.
    operator
        How the linear algebra is formed, a lowercase alphanumeric token, for
        example ``"dense"``, ``"sparse"`` or ``"na"`` (no operator, e.g. point
        sources).
    precision
        ``"fp64"``, ``"fp32"`` or ``"mp"`` (mixed precision).
    transformer
        The interferometer route, a lowercase alphanumeric token (``"wtilde"``,
        ``"nufft"``, ...), or ``None`` when there is none.
    rule_id
        What chose this setup: ``None`` for a setup written by hand,
        ``"explicit:PYAUTO_SETUP"`` when it came from the environment variable, or
        the Pulse decision id of the rule :meth:`auto` used.
    regime
        The dataset regime of the rule, e.g. ``"standard"``.
    hardware_class
        The hardware class of the rule, ``"cpu_many_core"`` or ``"gpu_a100"``.
    decision_id
        The PyAutoPulse decision id of the rule, if any.
    extrapolated
        Whether the rule was measured on different hardware from this machine (a
        non-A100 GPU, a CPU with fewer than ``MANY_CORE_MIN_CORES`` cores, or a
        hardware class the likelihood has no designation for).
    """

    backend: str
    device: str
    operator: str
    precision: str
    transformer: Optional[str] = None
    rule_id: Optional[str] = None
    regime: str = "standard"
    hardware_class: Optional[str] = None
    decision_id: Optional[str] = None
    extrapolated: bool = False

    def __post_init__(self):
        _validate_token("backend", self.backend, allowed=SETUP_BACKENDS)
        _validate_token("device", self.device, allowed=SETUP_DEVICES)
        _validate_token("operator", self.operator)
        _validate_token("precision", self.precision, allowed=SETUP_PRECISIONS)
        _validate_token("transformer", self.transformer, optional=True)

    @property
    def setup_id(self) -> str:
        """
        The setup_id ``<backend>_<device>_<operator>_<precision>[_<transformer>]``.
        """
        parts = [self.backend, self.device, self.operator, self.precision]
        if self.transformer is not None:
            parts.append(self.transformer)
        return "_".join(parts)

    @classmethod
    def from_setup_id(cls, setup_id: str, **provenance) -> "Setup":
        """
        The setup a setup_id names, e.g. ``Setup.from_setup_id("numba_cpu_sparse_fp64")``.

        The grammar is that of :func:`autonerves.environment.parse_setup_intent`;
        case and surrounding whitespace are ignored.

        Parameters
        ----------
        setup_id
            ``<backend>_<device>_<operator>_<precision>[_<transformer>]``.
        provenance
            Optional ``rule_id``, ``regime``, ``hardware_class``, ``decision_id``
            and ``extrapolated`` values for the returned setup.

        Raises
        ------
        ValueError
            If ``setup_id`` is not a setup_id (``"auto"``, ``"cpu"`` and ``"gpu"``
            are device intents, not setup_ids).
        """
        if not isinstance(setup_id, str):
            raise ValueError(f"A setup_id must be a string, got {setup_id!r}.")
        text = setup_id.strip().lower()
        if text in ("", "auto") + SETUP_DEVICES:
            raise ValueError(
                f"{setup_id!r} is a device intent, not a setup_id "
                "'<backend>_<device>_<operator>_<precision>[_<transformer>]'."
            )
        parse_setup_intent(text)
        backend, device, operator, precision, *rest = text.split("_")
        return cls(
            backend=backend,
            device=device,
            operator=operator,
            precision=precision,
            transformer=rest[0] if rest else None,
            **provenance,
        )

    @classmethod
    def auto(
        cls,
        dataset_type: str,
        variant: str,
        regime: str = "standard",
        environment: Optional[Environment] = None,
    ) -> "Setup":
        """
        The Pulse-designated optimal setup for a likelihood on this machine.

        In order:

        1. If ``PYAUTO_SETUP`` is a full setup_id, that setup is returned with
           ``rule_id="explicit:PYAUTO_SETUP"``. An explicit ``Setup(...)`` passed by
           the caller also always wins, by not calling this method.
        2. The hardware class is ``gpu_a100`` when ``PYAUTO_SETUP=gpu``, or when it
           is ``auto``/unset and the environment has a GPU; otherwise
           ``cpu_many_core``. The setup is marked ``extrapolated`` when the machine
           differs from the class the rule was measured on: a GPU whose name does
           not contain ``"A100"`` (or no GPU at all) for ``gpu_a100``, or fewer than
           ``MANY_CORE_MIN_CORES`` cores for ``cpu_many_core``.
        3. The rule ``(f"{dataset_type}/{variant}", regime, hardware_class)`` is
           looked up in the rule table. When the likelihood has no rule for that
           hardware class, its rule for the other class is used, with
           ``extrapolated=True``.

        Parameters
        ----------
        dataset_type
            The dataset type, e.g. ``"imaging"`` or ``"interferometer"``.
        variant
            The likelihood variant, e.g. ``"rectangular"``, ``"delaunay"`` or ``"mge"``.
        regime
            The dataset regime; only ``"standard"`` is designated so far.
        environment
            The hardware to choose for; defaults to ``Environment.detect()``.

        Raises
        ------
        ValueError
            If ``PYAUTO_SETUP`` is invalid, or the likelihood and regime have no
            rule; the message lists the known likelihoods.
        """
        value = os.environ.get(SETUP_ENV_VAR)
        intent = parse_setup_intent(value)
        if (
            value is not None
            and value.strip().lower() not in ("", "auto") + SETUP_DEVICES
        ):
            return cls.from_setup_id(value, rule_id=EXPLICIT_RULE_ID)

        environment = environment or Environment.detect()

        if intent == "auto":
            intent = "gpu" if environment.gpu else "cpu"

        if intent == "gpu":
            hardware_class = HARDWARE_CLASS_GPU
            extrapolated = not (
                environment.gpu and GPU_A100_NAME in (environment.gpu_name or "")
            )
        else:
            hardware_class = HARDWARE_CLASS_CPU
            extrapolated = environment.cores < MANY_CORE_MIN_CORES

        likelihood = f"{dataset_type}/{variant}"
        index = _rule_index()
        row = index.get((likelihood, regime, hardware_class))
        if row is None:
            other = (
                HARDWARE_CLASS_CPU
                if hardware_class == HARDWARE_CLASS_GPU
                else HARDWARE_CLASS_GPU
            )
            row = index.get((likelihood, regime, other))
            extrapolated = True
        if row is None:
            known = sorted({key[0] for key in index if key[1] == regime})
            raise ValueError(
                f"No setup rule for likelihood {likelihood!r} in regime {regime!r}. "
                f"Known likelihoods: {known}."
            )

        return cls.from_setup_id(
            row["optimal"],
            rule_id=row["decision_id"],
            regime=row["regime"],
            hardware_class=row["hardware_class"],
            decision_id=row["decision_id"],
            extrapolated=extrapolated,
        )
