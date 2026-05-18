import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Get an image
img_path = "105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima0_0.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Could not load image.")
    exit()

# Test 1: Original
faces1 = app.get(img)
print(f"Original image faces detected: {len(faces1)}")

# Test 2: Padded
pad_percent = 0.5
pad_h = int(img.shape[0] * pad_percent)
pad_w = int(img.shape[1] * pad_percent)
padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
faces2 = app.get(padded_img)
print(f"Padded image faces detected: {len(faces2)}")
