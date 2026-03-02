import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import collections

class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'C:/Pycharm_Program/Fall_Detection/models/fall_lstm_pytorch.pth'
YOLO_MODEL = YOLO('yolov8n-pose.pt')

model = FallLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

VIDEO_SOURCE = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/video_20260301_012001_fall.avi' 
cap = cv2.VideoCapture(0)

sequence_buffer = collections.deque(maxlen=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = YOLO_MODEL(frame, verbose=False)
    display_frame = frame.copy()

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        kp_xy = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()

        for i, (x, y) in enumerate(kp_xy):
            if kp_conf[i] > 0.5:
                cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        sequence_buffer.append(kp_xy.flatten())

        if len(sequence_buffer) == 30:
            input_data = torch.FloatTensor(np.array([list(sequence_buffer)])).to(DEVICE)
            
            with torch.no_grad():
                prediction = model(input_data)
                fall_prob = prediction.item()

            if fall_prob > 0.5:
                label = f"FALL! ({fall_prob*100:.1f}%)"
                color = (0, 0, 255)
            else:
                label = f"Normal ({(1-fall_prob)*100:.1f}%)"
                color = (0, 255, 0)

            cv2.rectangle(display_frame, (0, 0), (350, 60), (0, 0, 0), -1)
            cv2.putText(display_frame, label, (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Fall Detection Test", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()