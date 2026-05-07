import autoarray.plot as aplt

from pathlib import Path

import shutil

directory = Path(__file__).resolve().parent


def test__constructor():
    output = aplt.Output()

    assert output.path == None
    assert output._format == None
    assert output.format == "show"
    assert output.filename == None

    output = aplt.Output(path="Path", format="png", filename="file")

    assert output.path == "Path"
    assert output._format == "png"
    assert output.format == "png"
    assert output.filename == "file"

    if Path(output.path).exists():
        shutil.rmtree(output.path)


def test__input_path_is_created():
    test_path = str(directory / "files" / "output_path")

    if Path(test_path).exists():
        shutil.rmtree(test_path)

    assert not Path(test_path).exists()

    output = aplt.Output(
        path=test_path,
        format="png",
    )
    output.to_figure(structure=None, auto_filename="test")

    assert Path(test_path).exists()
