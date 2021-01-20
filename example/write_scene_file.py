import json
import numpy as np
from transform3d import Transform


def camera_matrix_from_fov(fovx, w, h):
    f = .5 * w / np.tan(fovx / 2)
    return np.array((
        (f, 0, w / 2),
        (0, f, h / 2),
        (0, 0, 1)
    ))


h, w = 480, 640
camera_matrix = camera_matrix_from_fov(np.deg2rad(45), w, h).tolist()

# read from blender
camera_poses = [
    Transform(p=(0., 0., 1.)),
    Transform(p=(1., 0., 0.5), rpy=(65, 0, 90), degrees=True),
    Transform(p=(0., 1., 0.), rpy=(90, 0, 180), degrees=True),
    Transform(p=(-1., -1., 0.5), rpy=(65, 0, -45), degrees=True),
    Transform(p=(0., -0.7, 0.6), rpy=(50, 0, 0), degrees=True),
]
# blender/opengl camera frame convention to cv2
camera_poses = [vp @ Transform(rpy=(np.pi, 0, 0)) for vp in camera_poses]

cameras = [dict(name='A', width=w, height=h, matrix=camera_matrix)]

objects = [
    dict(name='monkey', file_path='monkey.stl'),
    dict(name='torus', file_path='torus.stl'),
    dict(name='offset_cylinder', file_path='offset_cylinder.stl')
]

images = [dict(camera='A', file_path=f'{name}.png', camera_pose=vp.xyz_rotvec)
          for name, vp in zip('ABCDE', camera_poses)]

scene = dict(cameras=cameras, images=images, objects=objects)
json.dump(scene, open('scene.json', 'w'), indent=4)
