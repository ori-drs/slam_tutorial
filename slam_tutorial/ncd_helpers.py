import pathlib

# import gtsam
import pandas as pd


def load_ground_truth(path: pathlib.Path):
    df = pd.read_csv(
        str(path),
        skipinitialspace=True,
        names=["#sec", "nsec", "x", "y", "z", "qx", "qy", "qz", "qw"],
        comment="#",
    )
    return df
