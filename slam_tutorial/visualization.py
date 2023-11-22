import open3d as o3d
import slam_tutorial.colors as colors


def to_geometries(pose_graph):
    # Show nodes as poses
    viz_clouds = []
    node_centers = []
    for node in pose_graph.nodes:
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
    for e in pose_graph.edges:
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


def show_pose_graph(graph):
    geometries = to_geometries(graph)
    o3d.visualization.draw_geometries(
        geometries,
    )
