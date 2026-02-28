import cv2
import os
import collections

# 1. 설정
INPUT_VIDEO = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/video_20260301_012001_fall.avi'
SAVE_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

buffer_size = fps
frame_buffer = collections.deque(maxlen=buffer_size)

is_recording = False
video_writer = None
clip_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if is_recording:
        video_writer.write(frame)
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, "RECORDING CLIP...", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        frame_buffer.append(frame)

    cv2.imshow("Video Labeling Tool", display_frame)
    key = cv2.waitKey(int(1000/fps)) & 0xFF

    if key == ord('c'):
        if not is_recording:
            is_recording = True
            clip_count += 1
            file_path = os.path.join(SAVE_DIR, f"clip_{clip_count}.avi")
            video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            while frame_buffer:
                video_writer.write(frame_buffer.popleft())
        else:
            is_recording = False
            video_writer.release()
            frame_buffer.clear()

    elif key == ord('q'):
        break

if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()