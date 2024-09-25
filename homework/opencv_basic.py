#%%
import numpy as np
import cv2

img = cv2.imread("cat_image.jpeg")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

print(img.shape)

cv2.imshow("image", img)

cv2.waitKey(0)

cv2.imwrite("output.png", img)

cv2.destroyAllWindows

#%%
import cv2
import numpy as np

color = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)

print(color.shape)

height, width, channels = color.shape
cv2.imshow("Original Image",color)


b,g,r = cv2.split(color)

rgb_split = np.concatenate((b,g,r),axis=1)
cv2.imshow("BGR Channels", rgb_split)

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1)
cv2.imshow("Split HSV", hsv_split)

cv2.waitKey(0)

cv2.destroyAllWindows()

# %%
import cv2
import numpy as np

img = cv2.imread("strawberry.jpg")

cropped = img[50:450, 100:400]

resized = cv2.resize(cropped, (400,200))

rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Original", img)
cv2.imshow("Cropped Image", cropped)
cv2.imshow("Resized Image", resized)
cv2.imshow("Rotated Image", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
import cv2
import numpy as np

src = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)
src2 = cv2.imread("cat_image.jpeg", cv2.IMREAD_COLOR)
dst = cv2.bitwise_not(src)
src_resized = cv2.resize(src, (600, 400))
src2_resized = cv2.resize(src2, (600, 400))
dst1 = cv2.bitwise_and(src_resized, src2_resized)
dst2 = cv2.bitwise_or(src_resized, src2_resized)
dst3 = cv2.bitwise_xor(src_resized, src2_resized)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.imshow("dst3", dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

src = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()


#%%
import cv2
import numpy as np

src = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)
dst = cv2.blur(src, (3,3), anchor = (-1, -1), borderType=cv2.BORDER_DEFAULT)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()


#%%
import cv2
import numpy as np

src = cv2.imread("image/wheat.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
grad_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, 3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0.7)

laplacian = cv2.Laplacian(gray, cv2.CV_8U)

edges = cv2.Canny(src, threshold1=100, threshold2=200)

cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.imshow("edges", edges)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
import cv2
import numpy as np

src = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)
b,g,r = cv2.split(src)
inverse = cv2.merge((r,g,b))

zero = np.zeros_like(src, dtype=np.uint8)

b_merge = cv2.merge((b,zero,zero))
g_merge = cv2.merge((zero,g,zero))
r_merge = cv2.merge((zero,zero,r))

cv2.imshow("Original", src)
cv2.imshow("b", b_merge)
cv2.imshow("g", g_merge)
cv2.imshow("r", r_merge)
cv2.imshow("inverse", inverse)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
import numpy as np
import cv2
import time
import torch

cap = cv2.VideoCapture("ronaldingho.mp4")

# w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print("원본 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))

# resize_w = int(w / 2)
# resize_h = int(h / 2)
# print("변환된 동영상 너비(가로) : {}, 높이(세로) : {}".format(resize_w, resize_h))

image_counter = 0

print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp2/weights/best.pt', _verbose=False)
# model = YOLO('yolov5/runs/train/exp2/weights/best.pt')


while cap.isOpened():
    ret, frame = cap.read()

    if ret is False:
        cap.open("ronaldingho.mp4")
        continue
    
    result = model(frame)
    result.render()
    
    # resized_frame = cv2.resize(frame, (resize_w, resize_h))
    
    # cv2.imshow("Frame", resized_frame)
    cv2.imshow("Frame", frame)
    time.sleep(0.01)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    # elif key & 0xFF == ord('c'):
    #     print("C 키 입력 - 이미지 저장.")
    #     image_filename = f"saved_image_{image_counter}.png"
    #     cv2.imwrite(image_filename, frame)
    #     print(f"이미지 저장: {image_filename}")
    #     image_counter += 1
    elif key & 0xFF == ord('c'): 
        print("C 키 입력 - 사람 이미지 저장")
        
        for i, detection in enumerate(result.xyxy[0]):
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            person_img = frame[y_min:y_max, x_min:x_max]

            image_filename = f"person_{image_counter}.png"
            cv2.imwrite(image_filename, person_img)
            print(f"사람 감지 이미지 저장: {image_filename}")
            image_counter += 1
    
    
cap.release()
cv2.destroyAllWindows()

# %%
import numpy as np
import cv2
import time
import torch

cap = cv2.VideoCapture(0)

w = 640
h = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

image_counter = 0
video_counter = 0
record = False
print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp2/weights/best.pt', _verbose=False)

while cap.isOpened():
    ret, frame = cap.read()

    if ret is False:
        print("fail")
        break
    
    result = model(frame)
    result.render()
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        for i, detection in enumerate(result.xyxy[0]):
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            person_img = frame[y_min:y_max, x_min:x_max]

            image_filename = f"person_{image_counter}.png"
            cv2.imwrite(image_filename, person_img)
            print(f"사람 감지 이미지 저장: {image_filename}")
            image_counter += 1
    elif key & 0xFF == ord('v'):
        print("녹화 시작")
        record = True
        video_filename = f"video_{video_counter}.avi"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_filename, fourcc, fps, (w,h))
    elif key & 0xFF == ord('s'):
        print("녹화 중지")
        record = False
        video.release()
    if record == True:
        print("녹화중...")
        video.write(frame)
    
cap.release()
cv2.destroyAllWindows()

#%%
import numpy as np
import cv2
import time
import torch


points = []
color = (0, 255, 0)
radius = 20
bold = 0

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭
        points.append((x, y))  # 클릭한 위치 추가

def on_bold_trackbar(value):
    print("Trackbar value: ", value)
    global bold
    bold = value


cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_circle)
cv2.createTrackbar("bold", "Frame", bold, 10, on_bold_trackbar)

canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

top_left = (50, 50)
bottom_right = (300, 300)

w = 640
h = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

image_counter = 0
video_counter = 0
record = False

    
print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp2/weights/best.pt', _verbose=False)

while cap.isOpened():
    ret, frame = cap.read()

    cv2.line(frame, top_left, bottom_right, (0,255,0), 5)
    
    cv2.rectangle(frame, [pt + 30 for pt in top_left], [pt - 30 for pt in bottom_right], (0, 0, 255), 5)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me', [ pt + 80 for pt in top_left], font, 2, (0, 255, 255), 10)
    
    if ret is False:
        print("fail")
        break
    
    result = model(frame)
    result.render()
    
    for point in points:
        cv2.circle(frame, point, radius, color, -1)  # -1로 동그라미 채우기
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        for i, detection in enumerate(result.xyxy[0]):
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            person_img = frame[y_min:y_max, x_min:x_max]

            image_filename = f"person_{image_counter}.png"
            cv2.imwrite(image_filename, person_img)
            print(f"사람 감지 이미지 저장: {image_filename}")
            image_counter += 1
    elif key & 0xFF == ord('v'):
        print("녹화 시작")
        record = True
        video_filename = f"video_{video_counter}.avi"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_filename, fourcc, fps, (w,h))
    elif key & 0xFF == ord('s'):
        print("녹화 중지")
        record = False
        video.release()
    if record == True:
        print("녹화중...")
        video.write(frame)
    
cap.release()
cv2.destroyAllWindows()


#%%
import numpy as np
import cv2
import time
import torch

cap = cv2.VideoCapture("ronaldingho.mp4")

image_counter = 0

print("model load")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp2/weights/best.pt', _verbose=False)

while cap.isOpened():
    ret, frame = cap.read()

    if ret is False:
        cap.open("ronaldingho.mp4")
        continue
    
    result = model(frame)
    result.render()
    
    cv2.imshow("Frame", frame)
    time.sleep(0.01)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'): 
        print("C 키 입력 - 사람 이미지 저장")
        
        for i, detection in enumerate(result.xyxy[0]):
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            person_img = frame[y_min:y_max, x_min:x_max]

            image_filename = f"person_{image_counter}.png"
            cv2.imwrite(image_filename, person_img)
            print(f"사람 감지 이미지 저장: {image_filename}")
            image_counter += 1
    
cap.release()
cv2.destroyAllWindows()


#%%
