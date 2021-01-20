import os
from functools import partial
import argparse

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from scene import Scene, Object, Image
from editor import Editor


class ImageListItem(QtWidgets.QWidget):
    def __init__(self, image: Image):
        self.image = image
        self.selected = False
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.label = QtWidgets.QLabel(image.file_name)
        self.layout().addWidget(self.label)
        self.setAutoFillBackground(True)
        self.label.setStyleSheet("")
        self.update()
        image.open_listeners.add(lambda: self.set_selected(True))
        image.close_listeners.add(lambda: self.set_selected(False))

    def set_selected(self, selected):
        self.selected = selected
        self.update()

    def update(self):
        self.label.setStyleSheet('QLabel { color : ' + ('black' if self.selected else 'gray') + '; }')

    def mousePressEvent(self, a0: QtGui.QMouseEvent):
        self.image.toggle()


class ObjectListItem(QtWidgets.QWidget):
    def __init__(self, obj: Object):
        self.obj = obj
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout())
        label = QtWidgets.QLabel(obj.name)
        self.layout().addWidget(label)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.scene: Scene = None
        self.scene_fp: str = None

        h_layout = QtWidgets.QHBoxLayout(self)
        panel_widget = QtWidgets.QWidget()
        h_layout.addWidget(panel_widget)
        panel_layout = QtWidgets.QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_widget.setMaximumWidth(256)

        open_button = QtWidgets.QPushButton(text='open scene')
        open_button.clicked.connect(self.open_click)
        panel_layout.addWidget(open_button)
        self.save_button = QtWidgets.QPushButton(text='save pose')
        self.save_button.clicked.connect(self.save_pose)
        panel_layout.addWidget(self.save_button)
        self.pose_err_label = QtWidgets.QLabel()
        panel_layout.addWidget(self.pose_err_label)

        panel_layout.addWidget(QtWidgets.QLabel(text='Objects'))
        self.object_list = QtWidgets.QWidget()
        self.object_list.setLayout(QtWidgets.QVBoxLayout())
        self.object_list.layout().setSpacing(0)
        self.object_list.layout().setAlignment(Qt.AlignTop)
        panel_layout.addWidget(self.object_list)

        panel_layout.addWidget(QtWidgets.QLabel(text='Scene images (sorted by\nfarthest point sampling)'))
        self.image_list = QtWidgets.QWidget()
        self.image_list.setLayout(QtWidgets.QVBoxLayout())
        self.image_list.layout().setSpacing(0)
        self.image_list.layout().setAlignment(Qt.AlignTop)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.image_list)
        scroll.setWidgetResizable(True)
        panel_layout.addWidget(scroll)

        self.editor = Editor()
        h_layout.addWidget(self.editor)

    def save_pose(self):
        obj = self.editor.obj
        scene_dir = os.path.dirname(self.scene_fp)
        scene_name = '.'.join(os.path.basename(self.scene_fp).split('.')[:-1])
        name = '.'.join([scene_name, obj.name, 'pose'])
        obj.pose.save(os.path.join(scene_dir, name))
        i = (self.scene.objects.index(obj) + 1) % len(self.scene.objects)
        self.editor.select_object(self.scene.objects[i])

    def open_click(self):
        fp, _ = QtWidgets.QFileDialog().getOpenFileName()
        if fp:
            self.open_scene(fp)

    def clear(self):
        if self.scene:
            for image in self.scene.images:
                image.close()
            for c in self.image_list.children():
                if isinstance(c, ImageListItem):
                    c.deleteLater()
            for c in self.object_list.children():
                if isinstance(c, QtWidgets.QPushButton):
                    c.deleteLater()

    def open_scene(self, fp):
        self.clear()
        self.scene_fp = fp
        self.scene = Scene.from_file(fp)

        for image in self.scene.images:
            self.image_list.layout().addWidget(ImageListItem(image))

        for obj in self.scene.objects:
            item = QtWidgets.QPushButton(obj.name)
            item.clicked.connect(partial(self.editor.select_object, obj))
            self.object_list.layout().addWidget(item)

        self.editor.on_new_scene(self.scene)

        self.editor.select_object(self.scene.objects[0])
        for image in self.scene.images[:4]:
            image.open()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', nargs='?')
    scene_fp = parser.parse_args().scene

    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    if scene_fp:
        main_window.open_scene(scene_fp)
    main_window.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()
