import copy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

# import open3d.visualization.rendering as rendering


# colors
RED = [1, 0, 0]
GRAY = [0.5, 0.5, 0.5]


def show_pose_graph(
    graph,
    window_name="Pose Graph",
    show_ids=True,
    show_frames=True,
    show_edges=True,
    show_nodes=True,
    show_clouds=False,
    odometry_color=GRAY,
    loop_color=RED,
    up_to_node=np.inf,
):
    pose_graph = copy.deepcopy(graph)

    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(window_name, 1024, 768)
    vis.show_settings = True
    vis.show_skybox(False)

    node_centers = []
    frames_vis = o3d.geometry.TriangleMesh()
    nodes_vis = o3d.geometry.TriangleMesh()
    # clouds_vis = o3d.t.geometry.PointCloud()

    for n, node in enumerate(pose_graph.nodes):
        if n > up_to_node:
            break
        pose = node["pose"].matrix()
        pos = pose[0:3, 3]
        rot = pose[0:3, 0:3]
        node_centers.append(pos)

        if show_ids:
            vis.add_3d_label(pos + np.array([0.1, 0.1, 0.1]), f"{n}")

        if show_frames:
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=pos
            )
            frame_mesh.rotate(rot, center=pos)
            frames_vis += frame_mesh
            # vis.add_geometry(f"frame_{n}", frame_mesh)

        if show_nodes:
            node_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            node_mesh.translate(pos)
            nodes_vis += node_mesh
            # vis.add_geometry(f"node_{n}", frame_mesh)

        if show_clouds:
            cloud = pose_graph.get_node_cloud(n)
            cloud.transform(pose)
            # clouds_vis += cloud
            vis.add_geometry(f"cloud_{n}", cloud)

    vis.add_geometry("frames", frames_vis)
    vis.add_geometry("nodes", nodes_vis)
    # vis.add_geometry("clouds", clouds_vis)

    if show_edges:
        edges = []
        edge_colors = []
        for e in pose_graph.edges:
            edges.append([e["parent_id"], e["child_id"]])
            edge_colors.append(
                odometry_color if e["type"] == "odometry" else loop_color
            )

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(node_centers),
            lines=o3d.utility.Vector2iVector(edges),
        )
        line_set.colors = o3d.utility.Vector3dVector(edge_colors)
        vis.add_geometry("edges", line_set)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
