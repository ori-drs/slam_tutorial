# Author: Matias Mattamala (matias@robots.ox.ac.uk)

import numpy as np
import gtsam

from slam_tutorial.pose_graph import PoseGraph


def read_pose_gt(tokens):
    sec = tokens[0]
    nsec = tokens[1]
    pos = [float(i) for i in tokens[2:5]]
    q = [float(i) for i in tokens[5:9]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    pose = gtsam.Pose3(quat, pos)

    return pose, (sec, nsec)


def read_pose_slam(tokens):
    pose_id = int(tokens[0])
    pose_stamp = (tokens[8], tokens[9])
    pos = [float(i) for i in tokens[1:4]]
    q = [float(i) for i in tokens[4:8]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    pose = gtsam.Pose3(quat, pos)

    return pose, pose_stamp, pose_id


def read_pose_edge_slam(tokens):
    parent_id = int(tokens[0])
    child_id = int(tokens[1])
    pos = [float(i) for i in tokens[2:5]]
    q = [float(i) for i in tokens[5:9]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    relative_pose = gtsam.Pose3(quat, pos)
    upper_triangular = [float(i) for i in tokens[9:]]
    relative_info = np.eye(6, dtype=np.float64)
    relative_info[0, 0:6] = relative_info[0:6, 0] = upper_triangular[0:6]
    relative_info[1, 1:6] = relative_info[1:6, 1] = upper_triangular[6:11]
    relative_info[2, 2:6] = relative_info[2:6, 2] = upper_triangular[11:15]
    relative_info[3, 3:6] = relative_info[3:6, 3] = upper_triangular[15:18]
    relative_info[4, 4:6] = relative_info[4:6, 4] = upper_triangular[18:20]
    relative_info[5, 5:6] = relative_info[5:6, 5] = upper_triangular[20:21]

    return relative_pose, relative_info, parent_id, child_id


def load_ground_truth_file_as_pose_graph(
    path: str, distance_thr: float = 0.0, num_nodes: int = np.inf
):
    graph = PoseGraph()

    with open(path, "r") as file:
        lines = file.readlines()

        # Load nodes
        n = 0
        last_pose = None
        for line in lines:
            tokens = line.strip().split(",")
            if tokens[0][0] == "#":
                continue

            # Parse line
            pose, stamp = read_pose_gt(tokens)
            if last_pose is None:
                last_pose = pose
                continue

            # Just add nodes farther apart
            if (
                np.linalg.norm(pose.translation() - last_pose.translation())
                >= distance_thr
            ):
                graph.add_node(n, stamp, pose)
                last_pose = pose
                n += 1

            # Just add num_nodes nodes
            if n >= num_nodes:
                break

        # Add odometry edges
        for n, node in enumerate(graph.nodes):
            if n == 0:
                continue
            if n == graph.size:
                break

            ni = graph.nodes[n - 1]
            nj = graph.nodes[n]
            delta_pose = ni["pose"].inverse() * nj["pose"]
            graph.add_edge(n - 1, n, "odometry", delta_pose, np.eye(6) * 1000)

    # Return complete graph
    return graph


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
                pose, pose_stamp, pose_id = read_pose_slam(tokens[1:])
                graph.add_node(pose_id, pose_stamp, pose)

            elif tokens[0] == "EDGE_SE3:QUAT":
                relative_pose, relative_info, parent_id, child_id = read_pose_edge_slam(
                    tokens[1:]
                )
                edge_type = "odometry" if (parent_id == child_id - 1) else "loop"
                graph.add_edge(
                    parent_id, child_id, edge_type, relative_pose, relative_info
                )

    return graph


def write_graph_node(node):
    id = node["id"]
    sec, nsec = node["stamp"]
    x = node["pose"].translation()[0]
    y = node["pose"].translation()[1]
    z = node["pose"].translation()[2]

    qx = node["pose"].rotation().toQuaternion().x()
    qy = node["pose"].rotation().toQuaternion().y()
    qz = node["pose"].rotation().toQuaternion().z()
    qw = node["pose"].rotation().toQuaternion().w()

    stream = f"VERTEX_SE3:QUAT_TIME {id} {x} {y} {z} {qx} {qy} {qz} {qw} {sec} {nsec}\n"
    return stream


def write_graph_edge(edge):
    parent_id = edge["parent_id"]
    child_id = edge["child_id"]

    x = edge["pose"].translation()[0]
    y = edge["pose"].translation()[1]
    z = edge["pose"].translation()[2]

    qx = edge["pose"].rotation().toQuaternion().x()
    qy = edge["pose"].rotation().toQuaternion().y()
    qz = edge["pose"].rotation().toQuaternion().z()
    qw = edge["pose"].rotation().toQuaternion().w()

    info = ""
    for i in range(6):
        for j in range(i, 6):
            info += f"{edge['info'][i][j]} "
    info = info[:-1]

    stream = (
        f"EDGE_SE3:QUAT {parent_id} {child_id} {x} {y} {z} {qx} {qy} {qz} {qw} {info}\n"
    )
    return stream


def write_pose_graph(pose_graph: PoseGraph, path: str):
    with open(path, "w") as file:
        for node in pose_graph.nodes:
            file.write(write_graph_node(node))
        for edge in pose_graph.edges:
            file.write(write_graph_edge(edge))
