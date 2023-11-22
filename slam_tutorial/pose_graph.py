import gtsam
import numpy as np
import open3d as o3d


class PoseGraph:
    def __init__(self):
        self._nodes = []
        self._edges = []
        self._adjacency = {}
        self._clouds = {}

    @property
    def size(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @edges.setter
    def edges(self, edges):
        self._edges = edges

    def get_node_pose(self, id):
        return self._nodes[id]["pose"]

    def get_node_cloud(self, id):
        try:
            return self._clouds[id]
        except Exception:
            return o3d.t.geometry.PointCloud()

    def set_node_pose(self, id, pose):
        self._nodes[id]["pose"] = pose

    def get_odometry_edge(self, parent_id, child_id):
        try:
            edge_idx = self._adjacency[parent_id][child_id]
            return self._edges[edge_idx]["pose"]
        except Exception:
            return gtsam.Pose3.Identity()

    def _is_valid_id(self, id):
        return id >= 0 and id < self.size

    def add_node(self, id, stamp, pose):
        assert isinstance(pose, gtsam.Pose3)
        self._nodes.append({"pose": pose, "stamp": stamp, "id": id})
        self._adjacency[id] = {}

    def add_edge(self, parent_id, child_id, edge_type, relative_pose, relative_info):
        # This adds directed edges, even though they should be an undirected graph
        # We do this to simplify the API
        assert isinstance(relative_pose, gtsam.Pose3)
        assert isinstance(relative_info, np.ndarray)
        assert relative_info.shape == (6, 6)

        if not self._is_valid_id(parent_id):
            raise KeyError(
                f"Node parent [{parent_id}] not in graph. Cannot add the edge"
            )
        if not self._is_valid_id(child_id):
            raise KeyError(f"Node child [{child_id}] not in graph. Cannot add the edge")

        self._edges.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "pose": relative_pose,
                "info": relative_info,
                "type": edge_type,
            }
        )
        # Save reference to edge in adjacency matrix
        self._adjacency[parent_id][child_id] = len(self._edges) - 1

    def add_clouds(self, id, scan):
        assert isinstance(scan, o3d.t.geometry.PointCloud)
        self._clouds[id] = scan


def initialize_from_odometry(graph):
    node_initialized = [False] * graph.size
    node_initialized[0] = True

    for n, node in enumerate(graph.nodes):
        if node_initialized[n]:
            continue

        # Get previous pose
        last_pose = graph.nodes[n - 1]["pose"]

        # Get odometry from previous
        relative_pose = graph.get_odometry_edge(n - 1, n)
        pose = last_pose * relative_pose

        # Update current node pose
        graph.set_node_pose(n, pose)
        node_initialized[n] = True


def add_odometry_drift(
    graph, noise_per_m=0.0, axis="", rot_axis="", drift_type="constant"
):
    noise_mask = np.zeros(6)
    if "x" in axis:
        noise_mask[3] = 1.0
    if "y" in axis:
        noise_mask[4] = 1.0
    if "z" in axis:
        noise_mask[5] = 1.0

    if "x" in rot_axis:
        noise_mask[0] = 1.0
    if "y" in rot_axis:
        noise_mask[1] = 1.0
    if "z" in rot_axis:
        noise_mask[2] = 1.0

    import copy

    new_graph = copy.deepcopy(graph)
    for i, edge in enumerate(new_graph.edges):
        if edge["type"] != "odometry":
            continue

        # Get original odometry
        relative_pose = edge["pose"]

        # Drift model
        if drift_type == "constant":
            noise = np.ones(6)
        elif drift_type == "random_walk":
            noise = np.random.standard_normal(6)

        relative_distance = np.linalg.norm(relative_pose.translation())
        noise = noise * noise_mask * relative_distance * noise_per_m

        # Apply drift to odometry measurement
        new_relative_pose = relative_pose * gtsam.Pose3.Expmap(noise)

        # Update odometry measurement
        new_graph.edges[i]["pose"] = new_relative_pose

    initialize_from_odometry(new_graph)
    return new_graph


def create_test_pose_graph(noise_per_m=0.1, axis="xy", drift_type="random_walk"):
    import slam_tutorial.io as io
    import slam_tutorial

    # Read ground truth data
    graph_ground_truth = io.load_pose_graph(
        slam_tutorial.ASSETS_DIR + "/ground_truth.slam",
        clouds_path=slam_tutorial.ASSETS_DIR + "/individual_clouds",
    )

    # Create another one with odometry drift
    graph_with_drift = add_odometry_drift(
        graph_ground_truth, noise_per_m=noise_per_m, axis=axis, drift_type=drift_type
    )

    # Create a new pose graph with ground truth edges but drifting poses
    import copy

    graph = copy.deepcopy(graph_with_drift)
    graph.edges = graph_ground_truth.edges

    return graph
