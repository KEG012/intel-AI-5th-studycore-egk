#%%
from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 비디오 스트림 (카메라 입력)
cap = cv2.VideoCapture(0)

# 간단한 제스처 판별 함수
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]  # 엄지 끝
    index_tip = landmarks[8]  # 검지 끝
    middle_tip = landmarks[12]  # 중지 끝

    # 엄지와 검지 사이 거리 계산 (브이 제스처 예시)
    distance_thumb_index = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

    # 제스처 조건 (단순한 기준)
    if distance_thumb_index > 0.1 and index_tip.y < middle_tip.y:  # "브이" 제스처 (엄지와 검지가 떨어져 있음)
        return "V gesture"
    elif thumb_tip.y < index_tip.y:  # "엄지 업" 제스처
        return "Thumbs up"
    else:
        return "Unknown gesture"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8로 사람 탐지
    results = model(frame)

    # 결과의 바운딩 박스를 순회
    for result in results:
        for box in result.boxes:
            # 클래스 ID가 '사람'인 경우만 필터링 (YOLOv8의 사람 클래스는 보통 0번)
            if box.cls == 0:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)  # 바운딩 박스 좌표 추출
                person_roi = frame[y1:y2, x1:x2]  # 사람 바운딩 박스 내 영역 추출

                # 사람 ROI에서 손 인식
                rgb_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb_person_roi)

                # 손이 인식되면 손 랜드마크를 그립니다
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # MediaPipe의 손 랜드마크 그리기
                        mp.solutions.drawing_utils.draw_landmarks(
                            person_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # 손 랜드마크로 제스처 인식
                        landmarks = hand_landmarks.landmark
                        gesture = recognize_gesture(landmarks)
                        
                        # 제스처 결과를 화면에 출력
                        cv2.putText(person_roi, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 사람 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 결과 프레임을 보여줌
    cv2.imshow('YOLOv8 and Hand Gesture', frame)
    
    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
