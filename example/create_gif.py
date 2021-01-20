import cv2
import imageio
from tqdm import tqdm

cap = cv2.VideoCapture('pose_annotation_example.mov')
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale = 0.2
size = int(round(w * scale)), int(round(h * scale))

debug = False
images = []
i = 0

pb = tqdm(total=length)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    if i % 60 == 0:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        images.append(img)
    i += 1
    pb.update()
    if debug and i > 500:
        break
pb.close()

print('writing gif...')
imageio.mimsave('pose_annotation_example.gif', images, fps=5)
