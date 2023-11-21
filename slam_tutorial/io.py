# Author: Matias Mattamala (matias@robots.ox.ac.uk)

# import gtsam
import pandas as pd


def load_ground_truth(path: str):
    df = pd.read_csv(
        str(path),
        skipinitialspace=True,
        names=["#sec", "nsec", "x", "y", "z", "qx", "qy", "qz", "qw"],
        comment="#",
    )
    return df


def load_pose_graph(path: str):
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().split(" ")
            print(f"{type(tokens)} {tokens}")
