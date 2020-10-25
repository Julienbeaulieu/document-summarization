from dataclasses import dataclass


@dataclass
class GetPath:
    """
    This is a centralized class variable that stores all paths used.
    """

    from pathlib import Path
    import os

    root_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    data_dir = Path(os.path.join(root_dir, "dataset"))

    root: root_dir = root_dir
    data: data_dir = data_dir
    raw: Path = Path(root_dir / "raw")
    processed: Path = Path(data_dir / "processed")
    output: Path = Path(root_dir, "outputs")
    trained_weights: Path = Path(root_dir, "weights")


# This is THE Singleton that stores all path information.
AllPaths = GetPath()


def test_AllPaths():
    print(AllPaths)
    return AllPaths
