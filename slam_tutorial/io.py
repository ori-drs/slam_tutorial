# Author: Matias Mattamala (matias@robots.ox.ac.uk)

import numpy as np
import pandas as pd
from slam_tutorial.pose_graph import PoseGraph
from pytransform3d import transformations as pt


def load_ground_truth(path: str):
    df = pd.read_csv(
        str(path),
        skipinitialspace=True,
        names=["#sec", "nsec", "x", "y", "z", "qx", "qy", "qz", "qw"],
        comment="#",
    )
    return df

def parse_pose3(tokens: list):
    pose_id = int(tokens[0])
    pose_stamp = np.longdouble(tokens[8]) + 1e-9 * np.longdouble(tokens[9])
    pose = pt.transform_from_pq([float(i) for i in tokens[1:8]]).astype(dtype=np.float64)
    return pose_id, pose_stamp, pose

def parse_edge_pose3(tokens: list):
    parent_id = int(tokens[0])
    child_id = int(tokens[1])
    relative_pose = pt.transform_from_pq([float(i) for i in tokens[2:9]]).astype(dtype=np.float64)
    upper_triangular = [float(i) for i in tokens[9:]]
    relative_info = np.eye(6, dtype=np.float64)
    relative_info[0,0:6] = relative_info[0:6,0] = upper_triangular[0:6]
    relative_info[1,1:6] = relative_info[1:6,1] = upper_triangular[6:11]
    relative_info[2,2:6] = relative_info[2:6,2] = upper_triangular[11:15]
    relative_info[3,3:6] = relative_info[3:6,3] = upper_triangular[15:18]
    relative_info[4,4:6] = relative_info[4:6,4] = upper_triangular[18:20]
    relative_info[5,5:6] = relative_info[5:6,5] = upper_triangular[20:21]
    return parent_id, child_id, relative_pose, relative_info

def load_pose_graph(path: str):
    graph = PoseGraph()

    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().split(" ")
            
            # Parse lines
            if tokens[0] == "#":
                continue
            elif tokens[0] == "PLATFORM_ID":
                continue
            elif tokens[0] == "VERTEX_SE3:QUAT_TIME":
                pose_id, pose_stamp, pose = parse_pose3(tokens[1:])
                graph.add_node(pose_id, pose_stamp, pose)
            elif tokens[0] == "EDGE_SE3:QUAT":
                parent_id, child_id, relative_pose, relative_info = parse_edge_pose3(tokens[1:])
                edge_type = "odometry" if (parent_id == child_id - 1) else "loop_candidate"
                graph.add_edge(parent_id, child_id, edge_type, relative_pose, relative_info)
        
    return graph

            

