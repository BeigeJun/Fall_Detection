import cv2
import os
import collections

INPUT_VIDEO = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/video_20260308_221342_fall.avi'
BASE_SAVE_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video'
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

folder_map = {
    ord('1'): 'Stand Up',
    ord('2'): 'Stand Down',
    ord('3'): 'Sittin on Chair'
}

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

buffer_size = fps
frame_buffer = collections.deque(maxlen=buffer_size)

is_recording = False
video_writer = None
current_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if is_recording:
        video_writer.write(frame)
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, f"RECORDING TO: {current_label}", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        frame_buffer.append(frame)

    cv2.imshow("Video Labeling Tool", display_frame)
    key = cv2.waitKey(int(1000/fps)) & 0xFF

    if key in folder_map:
        if not is_recording:
            target_folder = folder_map[key]
            save_path = os.path.join(BASE_SAVE_DIR, target_folder)
            os.makedirs(save_path, exist_ok=True)
            
            clip_count = len(os.listdir(save_path)) + 1
            file_name = f"clip_{clip_count}.avi"
            full_path = os.path.join(save_path, file_name)
            
            video_writer = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
            is_recording = True
            current_label = target_folder
            
            while frame_buffer:
                video_writer.write(frame_buffer.popleft())
            print(f"Started recording: {full_path}")
            
        else:
            is_recording = False
            video_writer.release()
            frame_buffer.clear()
            print("Recording stopped.")

    elif key == ord('q'):
        break

if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()