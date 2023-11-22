import gtsam
import numpy as np
import open3d as o3d
import slam_tutorial.colors as colors


class PoseGraph:
    def __init__(self):
        self._nodes = []
        self._edges = []
        self._scans = {}

    @property
    def size(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def get_node_pose(self, id):
        return self._nodes[id]["pose"]

    def set_node_pose(self, id, pose):
        self._nodes[id]["pose"] = pose

    def _is_valid_id(self, id):
        return id >= 0 and id < self.size

    def add_node(self, id, stamp, pose):
        assert isinstance(pose, gtsam.Pose3)
        self._nodes.append({"pose": pose, "stamp": stamp, "id": id})

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

    def add_scans(self, id, scan):
        assert isinstance(scan, o3d.t.geometry.PointCloud)
        self._scans[id] = scan

    def to_viz(self, name="Pose Graph"):
        # Show nodes as poses
        viz_clouds = []
        node_centers = []
        for node in self._nodes:
            pose = node["pose"].matrix()
            pos = pose[0:3, 3]
            rot = pose[0:3, 0:3]
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=pos
            )
            frame_mesh.rotate(rot, center=pos)

            node_centers.append(pos)
            viz_clouds.append(frame_mesh)

        edges = []
        edge_colors = []
        for e in self._edges:
            edges.append([e["parent_id"], e["child_id"]])
            edge_colors.append(colors.GRAY if e["type"] == "odometry" else colors.RED)

        # Add edges
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(node_centers),
            lines=o3d.utility.Vector2iVector(edges),
        )
        line_set.colors = o3d.utility.Vector3dVector(edge_colors)
        viz_clouds.append(line_set)

        return viz_clouds

        # # o3d.visualization.draw_geometries(
        # #     viz_clouds,
        # #     window_name=name,
        # # )

        # o3d.visualization.draw(
        #     viz_clouds,
        #     show_skybox=False
        # )
