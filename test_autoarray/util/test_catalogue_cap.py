from autoarray.util import dataset_util


def test__cap_catalogue_size__inactive_without_env_var(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    assert dataset_util.cap_catalogue_size_for_small_datasets(200) == 200


def test__cap_catalogue_size__caps_when_active(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    assert dataset_util.cap_catalogue_size_for_small_datasets(200) == 25
    assert dataset_util.cap_catalogue_size_for_small_datasets(10) == 10
