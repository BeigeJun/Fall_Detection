import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import glob

VIDEO_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video'
SAVE_DIR = 'D:/3.Fall_Detection_DATA/2.Pos_CSV_DATA/processed_csv/fall'
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO('yolov8n-pose.pt')
CONF_THRESHOLD = 0.0

video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))

for video_path in video_files:
    video_name = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    all_sequences = [] 
    current_sequence = [] 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        display_frame = frame.copy()

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp_xy = results[0].keypoints.xy[0].cpu().numpy()
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()
            is_reliable = np.all(kp_conf >= CONF_THRESHOLD) and len(kp_xy) == 17

            if is_reliable:
                current_sequence.append(kp_xy.flatten())
                for x, y in kp_xy:
                    cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            else:
                current_sequence = []
                cv2.putText(display_frame, "LOW CONF - RESET", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            current_sequence = []

        if len(current_sequence) == 30:
            all_sequences.append(np.array(current_sequence))
            current_sequence = []

        cv2.putText(display_frame, f"Seqs: {len(all_sequences)} | Frames: {len(current_sequence)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Batch Extraction", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

    if all_sequences:
        final_data = []
        for seq_idx, seq in enumerate(all_sequences):
            for frame_idx, data in enumerate(seq):
                row = [seq_idx, frame_idx] + data.tolist()
                final_data.append(row)

        columns = ['seq_id', 'frame_id']
        for i in range(17):
            columns.extend([f'x{i}', f'y{i}'])

        df = pd.DataFrame(final_data, columns=columns)
        csv_name = video_name.replace('.avi', '.csv').replace('.AVI', '.csv')
        df.to_csv(os.path.join(SAVE_DIR, csv_name), index=False)
    else:
        pass

cv2.destroyAllWindows()