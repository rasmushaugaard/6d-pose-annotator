from functools import partial

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import numpy as np
from transform3d import Transform

from flow_layout import FlowLayout
from scene import Scene, Object, Image
from render_overlay import RenderOverlayWidget


class Editor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        vl = QtWidgets.QVBoxLayout(self)
        vl.setSpacing(0)
        vl.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        self.q = QtWidgets.QWidget()
        self.fl = FlowLayout(self.q)
        # scroll.setLayout(self.fl)
        scroll.setWidget(self.q)
        vl.addWidget(scroll)

        bottom_label = QtWidgets.QLabel(
            'Use mouse movement together with left click, ctrl/cmd and shift to define the pose.'
        )
        bottom_label.setAlignment(Qt.AlignBottom)
        bottom_label.setContentsMargins(5, 5, 5, 5)
        vl.addWidget(bottom_label)

        self.scene: Scene = None
        self.obj: Object = None
        self.overlays = []

    def open_image(self, image: Image):
        overlay = RenderOverlayWidget(image, self.crop_near)
        overlay.render_overlay.set_obj(self.obj)
        self.overlays.append(overlay)
        self.fl.addWidget(overlay)

    def crop_near(self):
        for overlay in self.overlays:
            overlay.render_overlay.crop_near_()

    def close_image(self, image: Image):
        for overlay in list(self.overlays):
            if overlay.image is image:
                overlay.remove()
                self.overlays.remove(overlay)

    def on_new_scene(self, scene: Scene):
        self.scene = scene
        for image in self.scene.images:
            image.open_listeners.add(partial(self.open_image, image))
            image.close_listeners.add(partial(self.close_image, image))

    def select_object(self, obj: Object):
        self.obj = obj
        for c in self.fl.children():
            c.remove()
        if obj.pose is None:
            open_images = [img for img in self.scene.images if img.is_open]
            if open_images:
                image = open_images[0]
            else:
                image = self.scene.images[0]

            obj.pose = image.camera_pose @ Transform(p=(0, 0, obj.diameter * 3), rpy=(np.pi / 2, 0, 0))
            obj.pose = obj.pose @ Transform(p=-obj.center)

        for overlay in self.overlays:
            overlay.render_overlay.set_obj(obj)
