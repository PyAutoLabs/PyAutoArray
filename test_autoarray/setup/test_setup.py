import os
from pathlib import Path

import pytest

import autoarray as aa
from autoarray.setup import generate_rules
from autoarray.setup import setup as setup_py
from autonerves.environment import Environment


def _environment(gpu=False, gpu_name=None, cores=64):
    return Environment(
        gpu=gpu,
        gpu_name=gpu_name,
        gpu_memory_gb=None,
        cores=cores,
        ram_gb=None,
        numba=True,
        nufftax=True,
        site="local",
    )


A100 = _environment(gpu=True, gpu_name="NVIDIA A100-SXM4-80GB")
MANY_CORE = _environment(cores=64)

RULES = setup_py.rules()


def _all_setup_ids():
    ids = set()
    for row in RULES:
        ids.add(row["optimal"])
        ids.update(row["secondary"])
    return sorted(ids)


@pytest.fixture(autouse=True)
def unset_pyauto_setup(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)


def test__setup__tokens_and_setup_id():
    setup = aa.Setup("jax", "gpu", "sparse", "fp64", transformer="wtilde")

    assert setup.setup_id == "jax_gpu_sparse_fp64_wtilde"
    assert setup.rule_id is None
    assert setup.regime == "standard"
    assert setup.extrapolated is False

    assert (
        aa.Setup("numba", "cpu", "sparse", "fp64").setup_id == "numba_cpu_sparse_fp64"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(backend="torch", device="cpu", operator="dense", precision="fp64"),
        dict(backend="jax", device="tpu", operator="dense", precision="fp64"),
        dict(backend="jax", device="cpu", operator="dense", precision="fp16"),
        dict(backend="jax", device="cpu", operator="Dense", precision="fp64"),
        dict(backend="jax", device="cpu", operator="den_se", precision="fp64"),
        dict(backend="jax", device="cpu", operator="", precision="fp64"),
        dict(
            backend="jax",
            device="cpu",
            operator="dense",
            precision="fp64",
            transformer="nu-fft",
        ),
    ],
)
def test__setup__invalid_tokens_raise(kwargs):
    with pytest.raises(ValueError):
        aa.Setup(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        "auto",
        "cpu",
        "gpu",
        "",
        "jax_cpu",
        "torch_cpu_dense_fp64",
        "jax_cpu_dense_fp64_a_b",
    ],
)
def test__from_setup_id__invalid_raises(value):
    with pytest.raises(ValueError):
        aa.Setup.from_setup_id(value)


def test__from_setup_id__case_and_whitespace():
    setup = aa.Setup.from_setup_id("  JAX_GPU_Dense_FP64_NUFFT ")

    assert setup.setup_id == "jax_gpu_dense_fp64_nufft"
    assert setup.transformer == "nufft"


@pytest.mark.parametrize("setup_id", _all_setup_ids())
def test__from_setup_id__round_trips_every_table_id(setup_id):
    assert aa.Setup.from_setup_id(setup_id).setup_id == setup_id


def test__rule_table__schema():
    assert len(RULES) == 39

    keys = [(r["likelihood"], r["regime"], r["hardware_class"]) for r in RULES]
    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys)

    for row in RULES:
        assert set(row) == set(generate_rules.ROW_KEYS)
        assert row["decision_id"]
        assert row["rule"]
        assert row["hardware_class"] in ("cpu_many_core", "gpu_a100")
        assert "/" in row["likelihood"]
        assert isinstance(row["secondary"], list)


@pytest.mark.parametrize(
    "row", RULES, ids=[f"{r['likelihood']}-{r['hardware_class']}" for r in RULES]
)
def test__auto__every_rule_row(row):
    dataset_type, variant = row["likelihood"].split("/")
    environment = A100 if row["hardware_class"] == "gpu_a100" else MANY_CORE

    setup = aa.Setup.auto(dataset_type, variant, environment=environment)

    assert setup.setup_id == row["optimal"]
    assert setup.rule_id == row["decision_id"]
    assert setup.decision_id == row["decision_id"]
    assert setup.hardware_class == row["hardware_class"]
    assert setup.regime == "standard"
    assert setup.extrapolated is False


def test__auto__examples():
    assert (
        aa.Setup.auto("imaging", "rectangular", environment=MANY_CORE).setup_id
        == "numba_cpu_sparse_fp64"
    )
    assert (
        aa.Setup.auto("imaging", "rectangular", environment=A100).setup_id
        == "jax_gpu_dense_fp64"
    )
    assert (
        aa.Setup.auto("interferometer", "delaunay", environment=A100).setup_id
        == "jax_gpu_sparse_fp64_wtilde"
    )
    assert (
        aa.Setup.auto("interferometer", "mge", environment=MANY_CORE).setup_id
        == "jax_cpu_dense_fp64_nufft"
    )


def test__auto__extrapolated_non_a100_gpu():
    environment = _environment(gpu=True, gpu_name="NVIDIA GeForce RTX 2060")

    setup = aa.Setup.auto("imaging", "delaunay", environment=environment)

    assert setup.hardware_class == "gpu_a100"
    assert setup.setup_id == "jax_gpu_dense_fp64"
    assert setup.extrapolated is True


def test__auto__extrapolated_gpu_without_name():
    setup = aa.Setup.auto(
        "imaging", "delaunay", environment=_environment(gpu=True, gpu_name=None)
    )

    assert setup.extrapolated is True


def test__auto__extrapolated_small_core_cpu():
    cores = setup_py.MANY_CORE_MIN_CORES - 1

    setup = aa.Setup.auto("imaging", "mge", environment=_environment(cores=cores))

    assert setup.hardware_class == "cpu_many_core"
    assert setup.setup_id == "jax_cpu_dense_fp64"
    assert setup.extrapolated is True

    setup = aa.Setup.auto(
        "imaging",
        "mge",
        environment=_environment(cores=setup_py.MANY_CORE_MIN_CORES),
    )
    assert setup.extrapolated is False


def test__auto__missing_hardware_class_uses_nearest_row_extrapolated():
    # datacube/delaunay is only designated on cpu_many_core.
    setup = aa.Setup.auto("datacube", "delaunay", environment=A100)

    assert setup.hardware_class == "cpu_many_core"
    assert setup.setup_id == "numba_cpu_sparse_fp64_wtilde"
    assert setup.extrapolated is True


def test__auto__unknown_likelihood_raises_listing_known():
    with pytest.raises(ValueError, match="imaging/rectangular"):
        aa.Setup.auto("imaging", "voronoi", environment=MANY_CORE)

    with pytest.raises(ValueError, match="Known likelihoods"):
        aa.Setup.auto("imaging", "rectangular", regime="huge", environment=MANY_CORE)


def test__auto__pyauto_setup_full_id_wins(monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "numpy_cpu_dense_fp64")

    setup = aa.Setup.auto("imaging", "rectangular", environment=A100)

    assert setup.setup_id == "numpy_cpu_dense_fp64"
    assert setup.rule_id == "explicit:PYAUTO_SETUP"
    assert setup.decision_id is None
    assert setup.hardware_class is None


def test__auto__pyauto_setup_full_id_wins_for_unknown_likelihood(monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "jax_gpu_sparse_fp64_wtilde")

    setup = aa.Setup.auto("anything", "else")

    assert setup.setup_id == "jax_gpu_sparse_fp64_wtilde"


def test__auto__pyauto_setup_cpu_forces_cpu_class(monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "cpu")

    setup = aa.Setup.auto(
        "imaging",
        "rectangular",
        environment=_environment(gpu=True, gpu_name="A100", cores=64),
    )

    assert setup.hardware_class == "cpu_many_core"
    assert setup.setup_id == "numba_cpu_sparse_fp64"
    assert setup.extrapolated is False


def test__auto__pyauto_setup_gpu_forces_gpu_class(monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "GPU")

    setup = aa.Setup.auto("imaging", "rectangular", environment=MANY_CORE)

    assert setup.hardware_class == "gpu_a100"
    assert setup.setup_id == "jax_gpu_dense_fp64"
    assert setup.extrapolated is True


@pytest.mark.parametrize("value", ["auto", "", " Auto "])
def test__auto__pyauto_setup_auto_detects(monkeypatch, value):
    monkeypatch.setenv("PYAUTO_SETUP", value)

    assert (
        aa.Setup.auto("imaging", "mge", environment=A100).hardware_class == "gpu_a100"
    )
    assert (
        aa.Setup.auto("imaging", "mge", environment=MANY_CORE).hardware_class
        == "cpu_many_core"
    )


def test__auto__pyauto_setup_invalid_raises(monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "tpu")

    with pytest.raises(ValueError, match="PYAUTO_SETUP"):
        aa.Setup.auto("imaging", "mge", environment=MANY_CORE)


def test__auto__detects_environment_when_not_given(monkeypatch):
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: A100))

    assert aa.Setup.auto("imaging", "mge").hardware_class == "gpu_a100"


def test__generate_rules__check_detects_drift(tmp_path):
    decisions = tmp_path / "pulse" / "decisions.yaml"
    decisions.parent.mkdir()
    decisions.write_text(
        "decisions:\n"
        "- id: imaging-mge-standard-cpu-many-core\n"
        "  kind: designation\n"
        "  likelihood: imaging/mge\n"
        "  regime: standard\n"
        "  hardware_class: cpu_many_core\n"
        "  optimal: jax_cpu_dense_fp64\n"
        "  secondary: []\n"
        "  rule: An imaging MGE fit uses jax_cpu_dense_fp64.\n"
        "- id: not-a-designation\n"
        "  title: something else\n"
    )
    output = tmp_path / "setup_rules.yaml"

    assert (
        generate_rules.main(["--pulse", str(decisions.parent), "--output", str(output)])
        == 0
    )
    assert (
        generate_rules.main(
            ["--pulse", str(decisions.parent), "--output", str(output), "--check"]
        )
        == 0
    )

    output.write_text(
        output.read_text().replace("jax_cpu_dense_fp64", "numba_cpu_sparse_fp64")
    )
    assert (
        generate_rules.main(
            ["--pulse", str(decisions.parent), "--output", str(output), "--check"]
        )
        == 1
    )


def _pulse_checkout():
    env = os.environ.get("PYAUTO_PULSE")
    if env and (Path(env) / "decisions.yaml").is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (
            parent / "organs" / "PyAutoPulse",
            parent / "PyAutoPulse",
        ):
            if (candidate / "decisions.yaml").is_file():
                return candidate
    return None


def test__rule_table__parity_with_pulse(tmp_path):
    pulse = _pulse_checkout()
    if pulse is None:
        pytest.skip(
            "No PyAutoPulse checkout (set PYAUTO_PULSE to run the parity test)."
        )

    output = tmp_path / "setup_rules.yaml"
    generate_rules.generate(pulse, output)

    regenerated = list(setup_py._load_rules(str(output)))

    assert regenerated == RULES
    assert generate_rules.check(pulse, setup_py.RULES_PATH)


class _FakeNumba:
    def __init__(self, max_threads=8, current=8, fail=False):
        self.config = type("config", (), {"NUMBA_NUM_THREADS": max_threads})
        self.current = current
        self.fail = fail
        self.calls = []

    def get_num_threads(self):
        return self.current

    def set_num_threads(self, n):
        if self.fail:
            raise RuntimeError("numba refused")
        self.calls.append(n)
        self.current = n


class _FakeThreadpoolctl:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def threadpool_limits(self, limits):
        if self.fail:
            raise RuntimeError("threadpoolctl refused")
        self.calls.append(limits)


def _patch_thread_modules(monkeypatch, numba=None, threadpoolctl=None):
    import sys

    monkeypatch.setitem(sys.modules, "numba", numba)
    monkeypatch.setitem(sys.modules, "threadpoolctl", threadpoolctl)


def test__apply_threads__sets_numba_and_threadpoolctl(monkeypatch):
    numba = _FakeNumba(max_threads=8, current=8)
    threadpoolctl = _FakeThreadpoolctl()
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=threadpoolctl)

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    assert setup.apply_threads(4) == 4
    assert numba.calls == [4]
    assert threadpoolctl.calls == [4]

    # Idempotent: numba is not reset when already at the count.
    assert setup.apply_threads(4) == 4
    assert numba.calls == [4]


def test__apply_threads__clips_to_numba_maximum_and_floors_at_one(monkeypatch):
    numba = _FakeNumba(max_threads=8, current=8)
    threadpoolctl = _FakeThreadpoolctl()
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=threadpoolctl)

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    assert setup.apply_threads(64) == 8
    assert threadpoolctl.calls == [64]

    assert setup.apply_threads(0) == 1
    assert numba.calls[-1] == 1


def test__apply_threads__default_from_environment(monkeypatch):
    numba = _FakeNumba(max_threads=16, current=16)
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=None)

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    assert setup.apply_threads() == 3

    monkeypatch.delenv("NUMBA_NUM_THREADS")
    monkeypatch.setattr("autonerves.environment.detect_cores", lambda: 5)
    assert setup.apply_threads() == 5


def test__apply_threads__threadpoolctl_missing(monkeypatch):
    numba = _FakeNumba(max_threads=8, current=8)
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=None)

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    assert setup.apply_threads(2) == 2
    assert numba.calls == [2]


def test__apply_threads__never_raises(monkeypatch, caplog):
    numba = _FakeNumba(max_threads=8, current=8, fail=True)
    threadpoolctl = _FakeThreadpoolctl(fail=True)
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=threadpoolctl)

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    assert setup.apply_threads(2) is None
    assert "numba" in caplog.text
    assert "BLAS" in caplog.text

    # Without numba, the threadpoolctl count is returned.
    _patch_thread_modules(
        monkeypatch, numba=None, threadpoolctl=_FakeThreadpoolctl()
    )
    assert setup.apply_threads(2) == 2

    # An unparseable count is logged, not raised.
    assert setup.apply_threads("many") is None


def test__apply_threads__real_numba():
    numba = pytest.importorskip("numba")

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    previous = numba.get_num_threads()
    try:
        assert setup.apply_threads(1) == 1
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(previous)


def test__setup_from__accepts_setup_setup_id_or_none():
    setup = aa.Setup.from_setup_id("jax_gpu_dense_fp64")

    assert setup_py.setup_from(setup) is setup
    assert setup_py.setup_from("jax_gpu_dense_fp64") == setup
    assert setup_py.setup_from(None) is None

    with pytest.raises(ValueError):
        setup_py.setup_from(3)


def test__apply_threads__worker_leaves_numba_alone(monkeypatch):
    numba = _FakeNumba(max_threads=8, current=8)
    threadpoolctl = _FakeThreadpoolctl()
    _patch_thread_modules(monkeypatch, numba=numba, threadpoolctl=threadpoolctl)
    monkeypatch.setattr(
        "autoarray.numba_util.parallel_kernel_allowed", lambda: False
    )

    setup = aa.Setup.from_setup_id("numba_cpu_sparse_fp64")

    assert setup.apply_threads(2) == 2
    assert numba.calls == []
    assert threadpoolctl.calls == [2]
