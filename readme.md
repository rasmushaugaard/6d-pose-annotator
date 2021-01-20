# 6d-pose-annotator

A graphical user interface for 6d pose estimation of objects in a scene. Here, a *scene* refers to a collection of images with unknown but static object poses and known camera intrinsics and extrinsics.


The scene has to be specified in a *json* file. 
See `example/scene.json` and `example/write_scene_file.py` for an example.
The tool assumes, the images have already been undistorted.
Object poses will be saved in the same folder as the scene file.

```$ pip install -r requirements.txt```

```$ python 6d_pose_annotator.py example/scene.json```

![pose-annotation-example](example/pose_annotation_example.gif)

