from enum import Enum

import numpy as np
import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from transform3d import Transform

import rendering
from scene import Image, Object
from rendering import _model_vertex_shader, _model_fragment_shader
import utils


class Mode(Enum):
    CROP = 'crop'
    ADJUST = 'adjust'


class RenderOverlay(QtOpenGL.QGLWidget):
    def __init__(self, image: Image, size=350):
        self.image = image
        self.obj: Object = None
        self.size = size

        self.mode = Mode.ADJUST
        self.last_mouse_position = None
        self.on_cam_t_model = lambda x: None
        self.overlay_hidden = False
        self.crop_start = None
        self.crop_mat = np.eye(4)
        self.ctx = None
        self.vao = None

        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        super().__init__(fmt, None)
        self.setFixedSize(size, size)
        self.set_mode(self.mode)
        self.reset_crop()

    def set_obj(self, obj: Object):
        if self.obj is not None:
            self.obj.pose_listeners.remove(self.update)
        self.obj = obj
        if self.obj is not None:
            obj.pose_listeners.add(self.update)
            if self.ctx:
                self.update_vbo()
        self.reset_crop()

    def update_vbo(self):
        if self.obj and self.ctx:
            with self.ctx:
                self.vbo.orphan(self.obj.vert_norms.size * 4)
                self.vbo.write(self.obj.vert_norms)
                self.vao.vertices = len(self.obj.mesh.faces) * 3
            self.update()

    @property
    def scale(self):
        return 1 / self.crop_mat[(0, 1), (0, 1)].max()

    @property
    def cam_t_model(self):
        if self.obj and self.obj.pose:
            return self.image.camera_pose.inv @ self.obj.pose
        else:
            return None

    @property
    def cam_t_model_gl(self):
        if self.cam_t_model:
            return utils.gl_t_cv @ self.cam_t_model
        else:
            return None

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        h, w = self.image.img.shape[:2]
        self.tex = self.ctx.texture((w, h), 3, self.image.img.data)
        self.tex.use()
        self.img_program = self.ctx.program(vertex_shader=rendering._image_vertex_shader,
                                            fragment_shader=rendering._image_fragment_shader)
        self.img_vao = self.ctx.simple_vertex_array(
            self.img_program, self.ctx.buffer(rendering._image_vert_tex), 'vert', 'tex_coord'
        )

        self.program = self.ctx.program(vertex_shader=_model_vertex_shader, fragment_shader=_model_fragment_shader)
        self.program['alpha'].value = .75

        self.vbo = self.ctx.buffer(self.obj.vert_norms if self.obj else [], dynamic=True)
        self.vao = self.ctx.simple_vertex_array(
            self.program, self.vbo, 'in_vert', 'in_norm'
        )

        self.line_prog = self.ctx.program(vertex_shader=rendering._line_vertex_shader,
                                          fragment_shader=rendering._line_fragment_shader)
        self.line_prog['color'].value = (0., 0., 0., 1.)
        self.line_buffer = self.ctx.buffer(np.zeros((4, 2), 'f4').reshape(-1))
        self.line_vao = self.ctx.simple_vertex_array(
            self.line_prog, self.line_buffer, 'in_vert'
        )

    def paintGL(self):
        # self.ctx.viewport = (0, 0, self.width(), self.height())
        self.ctx.clear()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.img_program['mvp'].value = utils.to_uniform(self.crop_mat)
        self.img_vao.render()
        # self.ctx.enable(moderngl.BLEND | moderngl.CULL_FACE)
        # self.ctx.blend_func = moderngl.SRC
        if self.cam_t_model and not self.overlay_hidden:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.program['mvp'].value = utils.to_uniform(
                self.crop_mat @ self.image.camera.projection_matrix @ self.cam_t_model_gl.matrix
            )
            self.program['mv'].value = utils.to_uniform(self.cam_t_model_gl.matrix)

            # The below makes sure only the nearest faces are drawn (which matters with transparency)
            # first, populate the depth buffer without drawing
            # then only draw depth equal
            self.ctx.enable(moderngl.BLEND)
            self.ctx.depth_func = '<'
            self.ctx.blend_func = moderngl.ZERO, moderngl.ONE
            self.vao.render()
            self.ctx.depth_func = '=='
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.vao.render()

        if self.crop_start:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.line_vao.render(moderngl.LINE_LOOP)

        self.ctx.finish()

    def set_mode(self, mode: Mode):
        self.mode = mode
        if mode == Mode.CROP:
            self.reset_crop()
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == ord(' '):
            self.overlay_hidden = True
            self.update()
        elif e.key() == ord('C'):
            self.set_mode(Mode.CROP)
        elif e.key() == ord('A'):
            self.set_mode(Mode.ADJUST)
        elif e.key() == ord('S'):
            self.crop_near()

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.key() == ord(' '):
            self.overlay_hidden = False
            self.update()

    def enterEvent(self, a0: QtCore.QEvent):
        self.setFocus()

    def leaveEvent(self, a0: QtCore.QEvent):
        self.clearFocus()
        self.overlay_hidden = False
        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        x, y = e.x(), e.y()
        if self.mode == Mode.CROP:
            self.crop_start = x, y
        self.last_mouse_position = x, y

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        x, y = e.x(), e.y()
        if self.mode == Mode.CROP:
            crop = np.clip((np.array((self.crop_start, (x, y))) + 0.5) / self.size, 0, 1)
            print(crop)
            if np.all(crop[0] == crop[1]):
                self.reset_crop()
            else:
                self.crop(crop * (2, -2) - 1)

    def crop(self, crop):  # crop in ndc coordinates relative to current crop
        scale = 2 / np.abs(crop[0] - crop[1]).max()
        offset = np.mean(crop, axis=0)
        mat = np.eye(4)
        mat[(0, 1), (0, 1)] = scale
        mat[:2, 3] = -offset * scale
        self.crop_mat = mat @ self.crop_mat
        self.crop_start = None
        self.set_mode(Mode.ADJUST)
        self.update()

    def crop_near_(self):
        self.reset_crop()
        mvp = self.crop_mat @ self.image.camera.projection_matrix @ self.cam_t_model_gl.matrix
        verts_ndc = utils.homogeneous(self.obj.mesh.vertices) @ mvp.T
        verts_ndc = verts_ndc[:, :2] / verts_ndc[:, 3:4]
        (x0, y0), (x1, y1) = verts_ndc.min(axis=0), verts_ndc.max(axis=0)
        crop_tight = np.array(((x0, y0), (x1, y1)))
        center = crop_tight.mean(axis=0)
        size = np.abs(crop_tight[0] - crop_tight[1])
        crop_near = np.array((center - size * .75, center + size * .75))
        self.crop(crop_near)

    def set_cam_t_model(self, cam_t_model: Transform):
        self.obj.set_pose(self.image.camera_pose @ cam_t_model)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        x_last, y_last = self.last_mouse_position
        x, y = e.x(), e.y()
        dx, dy = x - x_last, y - y_last
        self.last_mouse_position = x, y
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if self.mode == Mode.CROP:
            self.update_crop_drag_box(*self.crop_start, x, y)
            self.update()
        elif self.mode == Mode.ADJUST and self.obj and self.obj.pose:
            s = self.scale
            cam_t_center = self.cam_t_model @ Transform(p=self.obj.center)

            if modifiers == Qt.ControlModifier:  # move xy
                ss = self.cam_t_model.p[2] * 1e-3 * s
                self.set_cam_t_model(Transform(p=(dx * ss, dy * ss, 0)) @ self.cam_t_model)
            elif modifiers == (Qt.ControlModifier | Qt.ShiftModifier):  # move in depth
                dp = cam_t_center.p * dy * 2e-3 * s
                self.set_cam_t_model(Transform(p=dp) @ self.cam_t_model)
            elif modifiers == Qt.ShiftModifier:  # rotate cam z
                cam_R_model = Transform(rotvec=(0, 0, -dx * 3e-3)).R @ self.cam_t_model.R
                center_travel = Transform(p=self.cam_t_model.p, R=cam_R_model) @ self.obj.center - cam_t_center.p
                self.set_cam_t_model(Transform(p=self.cam_t_model.p - center_travel, R=cam_R_model))
            else:  # rotate cam x y
                rot = Transform(rotvec=(dy * 3e-3, -dx * 3e-3, 0))
                cam_R_model = rot.R @ self.cam_t_model.R
                center_travel = Transform(p=self.cam_t_model.p, R=cam_R_model) @ self.obj.center - cam_t_center.p
                self.set_cam_t_model(Transform(p=self.cam_t_model.p - center_travel, R=cam_R_model))

    def reset_crop(self):
        self.crop_start = None
        self.crop_mat = np.eye(4)
        h, w = self.image.img.shape[:2]
        aspect = w / h
        if aspect > 1:
            self.crop_mat[1, 1] = 1 / aspect
        else:
            self.crop_mat[0, 0] = aspect
        self.update()

    def update_crop_drag_box(self, x0, y0, x1, y1):
        line = (np.array(((x0, y0), (x0, y1), (x1, y1), (x1, y0))) + 0.5) / self.size
        line[:, 1] = 1 - line[:, 1]
        self.line_buffer.write(line.astype('f4').reshape(-1))


class RenderOverlayWidget(QtWidgets.QWidget):
    def __init__(self, image: Image, crop_near):
        super().__init__()
        self.image = image
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setAlignment(Qt.AlignTop)
        self.layout().setSpacing(0)
        self.render_overlay = RenderOverlay(image)
        self.render_overlay.crop_near = crop_near
        self.render_overlay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = QtWidgets.QToolBar()

        def create_button(text, cb=None):
            button = QtWidgets.QToolButton()
            button.setText(text)
            if cb is not None:
                button.clicked.connect(cb)
            else:
                button.setDisabled(True)
            self.toolbar.addWidget(button)

        create_button('adjust (a)', lambda _: self.render_overlay.set_mode(Mode.ADJUST))
        create_button('crop (c)', lambda _: self.render_overlay.set_mode(Mode.CROP))
        create_button('crop near (s)', lambda _: self.render_overlay.crop_near())

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred))
        self.toolbar.addWidget(spacer)
        label = QtWidgets.QLabel(image.file_name)
        label.setFont(QtGui.QFont('Arial', 10, italic=True))
        self.toolbar.addWidget(label)
        create_button('X', lambda _: self.image.close())

        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.render_overlay)

    def remove(self):
        if self.render_overlay.obj:
            self.render_overlay.obj.pose_listeners.remove(self.render_overlay.update)
        self.deleteLater()
