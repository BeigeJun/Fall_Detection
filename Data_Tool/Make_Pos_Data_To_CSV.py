import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import glob

# 1. 설정
VIDEO_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video/Stand Up'
SAVE_DIR = 'D:/3.Fall_Detection_DATA/2.Pos_CSV_DATA/processed_csv/Stand Up'
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO('yolov8n-pose.pt')
CONF_THRESHOLD = 0.5 

# [수정] 단순 기울기가 아닌, -180 ~ 180도 사이의 각도를 반환하는 함수
def get_angle(p1, p2):
    """
    p1(x, y)와 p2(x, y) 사이의 각도를 계산합니다.
    arctan2를 사용하면 dx=0인 경우(수직)도 자동으로 처리하며 
    값의 범위가 제한되어 신경망 학습에 매우 유리합니다.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    # 라디안을 구하고 degree로 변환 (-180 ~ 180)
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

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
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp_xy = results[0].keypoints.xy[0].cpu().numpy()
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()
            
            # 신뢰도가 너무 낮으면 해당 시퀀스는 건너뜀
            if len(kp_xy) == 17 and np.mean(kp_conf) >= CONF_THRESHOLD:
                # 12가지 주요 연결 부위의 각도(Angle) 계산
                angles = [
                    get_angle(kp_xy[5], kp_xy[6]),   # 0: 왼쪽 어깨 -> 오른쪽 어깨
                    get_angle(kp_xy[5], kp_xy[7]),   # 1: 왼쪽 어깨 -> 왼쪽 팔꿈치
                    get_angle(kp_xy[7], kp_xy[9]),   # 2: 왼쪽 팔꿈치 -> 왼쪽 손목
                    get_angle(kp_xy[6], kp_xy[8]),   # 3: 오른쪽 어깨 -> 오른쪽 팔꿈치
                    get_angle(kp_xy[8], kp_xy[10]),  # 4: 오른쪽 팔꿈치 -> 오른쪽 손목
                    get_angle(kp_xy[5], kp_xy[11]),  # 5: 왼쪽 어깨 -> 왼쪽 골반
                    get_angle(kp_xy[6], kp_xy[12]),  # 6: 오른쪽 어깨 -> 오른쪽 골반
                    get_angle(kp_xy[11], kp_xy[12]), # 7: 왼쪽 골반 -> 오른쪽 골반
                    get_angle(kp_xy[11], kp_xy[13]), # 8: 왼쪽 골반 -> 왼쪽 무릎
                    get_angle(kp_xy[13], kp_xy[15]), # 9: 왼쪽 무릎 -> 왼쪽 발목
                    get_angle(kp_xy[12], kp_xy[14]), # 10: 오른쪽 골반 -> 오른쪽 무릎
                    get_angle(kp_xy[14], kp_xy[16])  # 11: 오른쪽 무릎 -> 오른쪽 발목
                ]
                current_sequence.append(angles)
            else:
                current_sequence = [] 
        else:
            current_sequence = []

        if len(current_sequence) == 30:
            all_sequences.append(np.array(current_sequence))
            current_sequence = []

    cap.release()

    if all_sequences:
        final_data = []
        for seq_idx, seq in enumerate(all_sequences):
            for frame_idx, data in enumerate(seq):
                row = [seq_idx, frame_idx] + data.tolist()
                final_data.append(row)

        columns = [
            'seq_id', 'frame_id', 
            'sh_sh', 'l_sh_el', 'l_el_wr', 'r_sh_el', 'r_el_wr',
            'l_sh_hi', 'r_sh_hi', 'hi_hi', 'l_hi_kn', 'l_kn_an', 'r_hi_kn', 'r_kn_an'
        ]
        
        df = pd.DataFrame(final_data, columns=columns)
        csv_name = video_name.rsplit('.', 1)[0] + '_angles.csv'
        df.to_csv(os.path.join(SAVE_DIR, csv_name), index=False)
        print(f"정상 저장 완료: {csv_name}")

cv2.destroyAllWindows()


# import cv2
# from ultralytics import YOLO
# import pandas as pd
# import numpy as np
# import os
# import glob

# VIDEO_DIR = 'D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video'
# SAVE_DIR = 'D:/3.Fall_Detection_DATA/2.Pos_CSV_DATA/processed_csv/fall'
# os.makedirs(SAVE_DIR, exist_ok=True)

# model = YOLO('yolov8n-pose.pt')
# CONF_THRESHOLD = 0.0

# video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))

# for video_path in video_files:
#     video_name = os.path.basename(video_path)
    
#     cap = cv2.VideoCapture(video_path)
#     all_sequences = [] 
#     current_sequence = [] 

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break

#         results = model(frame, verbose=False)
#         display_frame = frame.copy()

#         if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
#             kp_xy = results[0].keypoints.xy[0].cpu().numpy()
#             kp_conf = results[0].keypoints.conf[0].cpu().numpy()
#             is_reliable = np.all(kp_conf >= CONF_THRESHOLD) and len(kp_xy) == 17

#             if is_reliable:
#                 current_sequence.append(kp_xy.flatten())
#                 for x, y in kp_xy:
#                     cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
#             else:
#                 current_sequence = []
#                 cv2.putText(display_frame, "LOW CONF - RESET", (10, 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             current_sequence = []

#         if len(current_sequence) == 30:
#             all_sequences.append(np.array(current_sequence))
#             current_sequence = []

#         cv2.putText(display_frame, f"Seqs: {len(all_sequences)} | Frames: {len(current_sequence)}", 
#                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#         cv2.imshow("Batch Extraction", display_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit()

#     cap.release()

#     if all_sequences:
#         final_data = []
#         for seq_idx, seq in enumerate(all_sequences):
#             for frame_idx, data in enumerate(seq):
#                 row = [seq_idx, frame_idx] + data.tolist()
#                 final_data.append(row)

#         columns = ['seq_id', 'frame_id']
#         for i in range(17):
#             columns.extend([f'x{i}', f'y{i}'])

#         df = pd.DataFrame(final_data, columns=columns)
#         csv_name = video_name.replace('.avi', '.csv').replace('.AVI', '.csv')
#         df.to_csv(os.path.join(SAVE_DIR, csv_name), index=False)
#     else:
#         pass

# cv2.destroyAllWindows()