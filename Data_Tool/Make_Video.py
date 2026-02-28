import cv2
import time
import os

SAVE_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save'
os.makedirs(SAVE_DIR, exist_ok=True)

label = ['normal','fall']

choose_label = 1

timestamp = time.strftime("%Y%m%d_%H%M%S")
file_name = f"video_{timestamp}_{label[choose_label]}.avi"
full_path = os.path.join(SAVE_DIR, file_name)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fps = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID') # AVI 포맷용 코덱

out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.putText(frame, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Recording...', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()