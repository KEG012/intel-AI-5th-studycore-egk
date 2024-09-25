#%%
import torch
import cv2
from ultralytics import YOLO

print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', _verbose=False)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)

print("image load")
im2 = cv2.imread("cat_image2.jpeg")
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
results = model(im2_rgb)
results.save()
pred = results.pandas().xyxy[0]
predNP = pred.to_numpy()
print(pred)
print(predNP)
nj, ni = predNP.shape

for n, i in enumerate(pred.columns):
    #print(n, i)
    if i == "name":
        #print(((predNP.shape)))
        for j in predNP[:, n]:
            if j == 'Cat':
               print("Find!")
               
#%%

import torch
import cv2
from ultralytics import YOLO

print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt', _verbose=False)
# model = YOLO('runs/train/exp2/weights/best.pt')

print("image load")
cap = cv2.VideoCapture()

im2 = cv2.imread("people.jpeg")
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
results = model(im2_rgb)
results.save()
pred = results.pandas().xyxy[0]
predNP = pred.to_numpy()
print(pred)
print(predNP)
nj, ni = predNP.shape

for n, i in enumerate(pred.columns):
    #print(n, i)
    if i == "name":
        #print(((predNP.shape)))
        for j in predNP[:, n]:
            if j == 'person':
               print("Find!")

