import open3d as o3d
import numpy as np

# hello world program for open3d

points = np.random.uniform(-1, 1, size=(1000, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd.paint_uniform_color([0.1, 0.7, 0.9])

o3d.visualization.draw_geometries([pcd])
