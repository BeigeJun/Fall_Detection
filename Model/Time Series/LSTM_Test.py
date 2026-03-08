import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import collections
import time

# 1. 모델 정의 (학습 사양: 128, 3층)
class DeepFallLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3):
        super(DeepFallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.3, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

def get_angle(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.degrees(np.arctan2(dy, dx))

# 2. 설정 및 모델 로딩
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'C:/Pycharm_Program/Fall_Detection/models/deep_fall_lstm.pth'
YOLO_MODEL = YOLO('yolov8n-pose.pt')
LABEL_NAMES = ['Normal', 'Fall', 'Stand Up', 'Stand Down', 'Sitting']

model = DeepFallLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. 영상 소스 및 속도 설정
video_path = "D:/3.Fall_Detection_DATA/1.Pos_Video_Save/video_20260308_221342_fall.avi"
cap = cv2.VideoCapture(0)

# 원본 영상의 FPS를 가져와서 프레임당 지연 시간(ms) 계산
original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps == 0 or original_fps > 100: original_fps = 30.0
frame_delay = 1000 / original_fps  # 예: 30fps라면 33.3ms

sequence_buffer = collections.deque(maxlen=30)

print(f"영상 재생 시작 (원본 FPS: {original_fps:.1f})")

while cap.isOpened():
    start_time = time.time()  # 연산 시작 시간 측정
    
    ret, frame = cap.read()
    if not ret: break

    results = YOLO_MODEL(frame, verbose=False)
    display_frame = frame.copy()

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        kp_xy = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()

        if len(kp_xy) == 17 and np.mean(kp_conf) > 0.5:
            angles = [
                get_angle(kp_xy[5], kp_xy[6]), get_angle(kp_xy[5], kp_xy[7]),
                get_angle(kp_xy[7], kp_xy[9]), get_angle(kp_xy[6], kp_xy[8]),
                get_angle(kp_xy[8], kp_xy[10]), get_angle(kp_xy[5], kp_xy[11]),
                get_angle(kp_xy[6], kp_xy[12]), get_angle(kp_xy[11], kp_xy[12]),
                get_angle(kp_xy[11], kp_xy[13]), get_angle(kp_xy[13], kp_xy[15]),
                get_angle(kp_xy[12], kp_xy[14]), get_angle(kp_xy[14], kp_xy[16])
            ]
            sequence_buffer.append(np.array(angles) / 180.0)

            for x, y in kp_xy[5:]:
                cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        if len(sequence_buffer) == 30:
            input_data = torch.FloatTensor(np.array([list(sequence_buffer)])).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_data)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
                
                label = LABEL_NAMES[predicted_idx.item()]
                confidence = max_prob.item() * 100

            color = (0, 0, 255) if label == 'Fall' else (0, 255, 0)
            cv2.rectangle(display_frame, (0, 0), (550, 65), (0, 0, 0), -1)
            cv2.putText(display_frame, f"{label} ({confidence:.1f}%)", (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Multi-Class Fall Detection", display_frame)

    # --- 실시간 속도 조절 핵심 로직 ---
    # 연산에 소요된 시간(ms) 계산
    processing_time = (time.time() - start_time) * 1000
    # 원본 지연 시간에서 연산 시간을 뺀 만큼만 대기
    wait_time = max(1, int(frame_delay - processing_time))
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# import torch
# import torch.nn as nn
# import numpy as np
# from ultralytics import YOLO
# import collections

# class FallLSTM(nn.Module):
#     def __init__(self, input_size=34, hidden_size=64, num_layers=2):
#         super(FallLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         out = self.fc(hn[-1])
#         return out

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_PATH = 'C:/Pycharm_Program/Fall_Detection/models/fall_lstm_pytorch.pth'
# YOLO_MODEL = YOLO('yolov8n-pose.pt')

# model = FallLSTM().to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()

# VIDEO_SOURCE = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/video_20260301_012001_fall.avi' 
# cap = cv2.VideoCapture(0)

# sequence_buffer = collections.deque(maxlen=30)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break

#     results = YOLO_MODEL(frame, verbose=False)
#     display_frame = frame.copy()

#     if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
#         kp_xy = results[0].keypoints.xy[0].cpu().numpy()
#         kp_conf = results[0].keypoints.conf[0].cpu().numpy()

#         for i, (x, y) in enumerate(kp_xy):
#             if kp_conf[i] > 0.5:
#                 cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

#         sequence_buffer.append(kp_xy.flatten())

#         if len(sequence_buffer) == 30:
#             input_data = torch.FloatTensor(np.array([list(sequence_buffer)])).to(DEVICE)
            
#             with torch.no_grad():
#                 prediction = model(input_data)
#                 fall_prob = prediction.item()

#             if fall_prob > 0.5:
#                 label = f"FALL! ({fall_prob*100:.1f}%)"
#                 color = (0, 0, 255)
#             else:
#                 label = f"Normal ({(1-fall_prob)*100:.1f}%)"
#                 color = (0, 255, 0)

#             cv2.rectangle(display_frame, (0, 0), (350, 60), (0, 0, 0), -1)
#             cv2.putText(display_frame, label, (10, 45), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

#     cv2.imshow("Fall Detection Test", display_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()