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
            return o3d.geometry.PointCloud()

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
        assert isinstance(scan, o3d.geometry.PointCloud)
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
    graph,
    noise_per_m=0.0,
    axis="",
    rot_axis="",
    drift_type="constant",
    reset_node_poses=True,
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

    if reset_node_poses:
        initialize_from_odometry(new_graph)
    return new_graph


def create_test_pose_graph(
    use_wrong_init=True,
    use_noisy_odom=False,
    use_true_loops=False,
    use_false_loops=False,
    init_noise_per_m=0.1,
    init_drift_axis="xy",
    init_drift_type="random_walk",
    odo_noise_per_m=0.1,
    odo_drift_axis="xy",
    odo_drift_type="random_walk",
    loop_candidate_id_distance=10,
    true_loop_dist_thr=5,
    false_loop_dist_thr=10,
    true_loop_num=2,
    false_loop_num=10,
    loop_info=1000,
):
    import slam_tutorial.io as io
    import slam_tutorial
    import copy

    # Read ground truth data
    graph_ground_truth = io.load_pose_graph(
        slam_tutorial.ASSETS_DIR + "/ground_truth.slam",
        clouds_path=slam_tutorial.ASSETS_DIR + "/individual_clouds",
    )
    out_graph = copy.deepcopy(graph_ground_truth)

    # Create a graph with odometry drift
    if use_wrong_init:
        graph_init = add_odometry_drift(
            graph_ground_truth,
            noise_per_m=init_noise_per_m,
            axis=init_drift_axis,
            drift_type=init_drift_type,
            reset_node_poses=True,
        )
        out_graph.nodes = graph_init.nodes

    if use_noisy_odom:
        graph_odo = add_odometry_drift(
            graph_ground_truth,
            noise_per_m=odo_noise_per_m,
            axis=odo_drift_axis,
            drift_type=odo_drift_type,
            reset_node_poses=False,
        )
        out_graph.edges = graph_odo.edges

    if use_true_loops or use_false_loops:
        import random

        true_loop_candidates = []
        false_loop_candidates = []
        for i in range(graph_ground_truth.size):
            for j in range(i, graph_ground_truth.size):
                if abs(i - j) > loop_candidate_id_distance:
                    posei = graph_ground_truth.get_node_pose(i)
                    posej = graph_ground_truth.get_node_pose(j)
                    relative_pose = posei.inverse() * posej
                    relative_distance = np.linalg.norm(relative_pose.translation())

                    if relative_distance < true_loop_dist_thr:
                        true_loop_candidates.append(
                            {"parent_id": i, "child_id": j, "pose": relative_pose}
                        )
                    if relative_distance > false_loop_dist_thr:
                        eye = gtsam.Pose3.Identity()
                        false_loop_candidates.append(
                            {"parent_id": i, "child_id": j, "pose": eye}
                        )

        # Add true loop candidates
        if use_true_loops:
            random.shuffle(true_loop_candidates)
            for c in true_loop_candidates[:true_loop_num]:
                out_graph.add_edge(
                    c["parent_id"],
                    c["child_id"],
                    "loop",
                    c["pose"],
                    loop_info * np.eye(6),
                )

        # Add false loop candidates
        if use_false_loops:
            random.shuffle(false_loop_candidates)
            for c in false_loop_candidates[:false_loop_num]:
                out_graph.add_edge(
                    c["parent_id"],
                    c["child_id"],
                    "loop",
                    c["pose"],
                    loop_info * np.eye(6),
                )

    return out_graph
