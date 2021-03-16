import os
import json
from typing import List, Dict
from functools import lru_cache

import cv2
import numpy as np
import trimesh
from transform3d import Transform

import utils


class Camera:
    def __init__(self, matrix: np.ndarray, width: int, height: int):
        assert matrix.shape == (3, 3)
        self.matrix = matrix
        self.width = width
        self.height = height
        self.projection_matrix = utils.projection_matrix_from_camera_matrix(self.matrix, self.height, self.width)


class Image:
    def __init__(self, file_path: str, camera: Camera, camera_pose: Transform):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.camera = camera
        self.camera_pose = camera_pose
        self.is_open = False
        self.open_listeners = set()
        self.close_listeners = set()

    @property
    @lru_cache(1)
    def img(self):
        return cv2.imread(self.file_path, cv2.IMREAD_COLOR)[..., ::-1].copy()

    def open(self):
        self.is_open = True
        for cb in self.open_listeners:
            cb()

    def close(self):
        self.is_open = False
        for cb in self.close_listeners:
            cb()

    def toggle(self):
        if self.is_open:
            self.close()
        else:
            self.open()


class Object:
    def __init__(self, name: str, file_path: str, pose: Transform = None):
        self.name = name
        self.file_path = file_path
        self.pose = pose
        self.pose_listeners = set()

    def set_pose(self, pose: Transform):
        self.pose = pose
        for cb in self.pose_listeners:
            cb()

    @property
    @lru_cache(1)
    def mesh(self) -> trimesh.Trimesh:
        return trimesh.load_mesh(self.file_path)

    @property
    @lru_cache(1)
    def vert_norms(self):
        return utils.mesh_to_vertnorms(self.mesh)

    @property
    @lru_cache(1)
    def center(self):
        return self.mesh.bounding_sphere.primitive.center

    @property
    @lru_cache(1)
    def diameter(self):
        verts = self.mesh.convex_hull.vertices
        return np.linalg.norm(verts[None] - verts[:, None], axis=-1).max()


class Scene:
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.images: List[Image] = []
        self.objects: List[Object] = []

    @classmethod
    def from_file(cls, fp):
        fp_dir = os.path.dirname(fp)
        scene = cls()
        data = json.load(open(fp))
        for d in data['cameras']:
            scene.cameras[d['name']] = Camera(np.array(d['matrix']), d['width'], d['height'])
        for d in data['objects']:
            scene.objects.append(Object(
                d['name'],
                os.path.join(fp_dir, d['file_path']),
                Transform.from_xyz_rotvec(d['pose']) if 'pose' in d else None,
            ))
        images = []
        for d in data['images']:
            images.append(Image(
                os.path.join(fp_dir, d['file_path']),
                scene.cameras[d['camera']],
                Transform.from_xyz_rotvec(d['camera_pose'])
            ))
        order = utils.farthest_point_sampling(np.array([img.camera_pose.p for img in images]), len(images))
        for i in order:
            scene.images.append(images[i])
        return scene
