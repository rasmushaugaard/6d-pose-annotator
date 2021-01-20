import numpy as np
import trimesh
from PyQt5 import QtGui, QtWidgets
from transform3d import Transform


def homogeneous(points: np.ndarray):  # (N, d) -> (N, d+1)
    n, d = points.shape
    return np.concatenate((points, np.ones((n, 1))), axis=1)


def orthographic_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float):
    return np.array((
        (2 / (right - left), 0, 0, -(right + left) / (right - left)),
        (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
        (0, 0, -2 / (far - near), -(far + near) / (far - near)),
        (0, 0, 0, 1),
    ))


def projection_matrix_from_camera_matrix(camera_matrix, h, w, near=0.1, far=10):
    # http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[:2, :3] = cvt_camera_matrix_cv_gl(camera_matrix[:2, :3])
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    orth = orthographic_matrix(0, w, 0, h, near, far)
    return orth @ persp


def camera_matrix_from_fov(fov, res):
    # projects from cv2 camera frame to image frame
    f = res / np.tan(fov / 2) / 2
    cx = cy = res / 2 - .5
    return np.array((
        (f, 0, cx),
        (0, f, cy),
        (0, 0, 1)
    ))


def mesh_to_vertnorms(mesh: trimesh.Trimesh):
    verts = mesh.vertices[mesh.faces].reshape(-1, 3)
    norms = np.tile(mesh.face_normals[:, None], (1, 3, 1)).reshape(-1, 3)
    # norms = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
    return np.concatenate((verts, norms), -1).reshape(-1).astype('f4')


def to_uniform(a):
    a = np.asarray(a, 'f4')
    if a.ndim == 2:
        a = a.T.reshape(-1)
    assert a.ndim == 1
    return tuple(a)


gl_t_cv = Transform(rpy=(np.pi, 0, 0))


def cvt_camera_matrix_cv_gl(camera_matrix: np.ndarray):
    camera_matrix_out = camera_matrix.copy()
    camera_matrix_out[:, 2] *= -1
    return camera_matrix_out


def farthest_point_sampling(all_points: np.ndarray, n: int):  # (N, d)
    assert n >= 0
    assert all_points.ndim == 2
    sampled_points_idx = []
    sampled_points = all_points.mean(axis=0, keepdims=True)
    for _ in range(n):
        dists = np.linalg.norm(all_points[:, None] - sampled_points[None], axis=-1)  # (N, n_cur)
        scores = dists.min(axis=1)  # (N,)
        scores[sampled_points_idx] = 0
        sampled_points_idx.append(np.argmax(scores))
        sampled_points = all_points[sampled_points_idx]
    return sampled_points_idx


def _farthest_point_sampling_test():
    a = np.array((
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (1, 1, 1),
        (2, -2, 3)
    ))
    assert farthest_point_sampling(a, 3) == [3, 0, 2]


def qimg(img):
    h, w, c = img.shape
    if c == 3:
        f = QtGui.QImage.Format_RGB888
    elif c == 4:
        f = QtGui.QImage.Format_RGBA8888
    else:
        raise ValueError()
    return QtGui.QImage(img.data, w, h, c * w, f)


class Image(QtWidgets.QLabel):
    def __init__(self, img=None, **kwargs):
        super().__init__(**kwargs)
        if img is not None:
            self.update_img(img)

    def update_img(self, img):
        self.setPixmap(QtGui.QPixmap(qimg(img)))


if __name__ == '__main__':
    _farthest_point_sampling_test()
