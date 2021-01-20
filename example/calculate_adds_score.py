import numpy as np
from scipy.spatial.kdtree import KDTree
from scipy.spatial.ckdtree import cKDTree
import trimesh
from transform3d import Transform

poses_gt = [
    ('monkey', Transform(rpy=(15, 30, 45), degrees=True)),
    ('torus', Transform(p=(-0.05, 0.07, 0), rpy=(-140, -100, -80), degrees=True)),
    ('offset_cylinder', Transform(p=(-0.3, -0.05, 0.05), rpy=(-128, -15, 35), degrees=True)),
]

for name, pose_gt in poses_gt:
    pose_est = Transform.load(f'scene.{name}.pose')

    mesh = trimesh.load_mesh(f'{name}.stl')
    hv = mesh.convex_hull.vertices
    diameter = np.linalg.norm(hv[None] - hv[:, None], axis=-1).max()

    v_gt = pose_gt @ mesh.vertices
    v_est = pose_est @ mesh.vertices

    tree_gt = cKDTree(v_gt)  # type: KDTree
    dists, _ = tree_gt.query(v_est)
    adds_score = np.mean(dists) / diameter
    print(f'{name}: {adds_score:.3f}')
