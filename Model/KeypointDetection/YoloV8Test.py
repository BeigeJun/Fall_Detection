import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture('D:/3.Fall_Detection_DATA/1.Pos_Video_Save/Cutting_Fall_Video/clip_1.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        
        kp_xy = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()

        visible_points = kp_xy[kp_xy[:, 0] > 0]
        
        if len(visible_points) == 17:
            for i in range(17):
                if(kp_conf[i] > 0.7):
                    x, y = kp_xy[i]

                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(x), int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        else:
            pass

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()