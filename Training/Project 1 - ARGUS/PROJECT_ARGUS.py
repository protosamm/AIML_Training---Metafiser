from ultralytics import YOLO
import cv2
from math import dist

# Load models
pose_model = YOLO('yolov8n-pose.pt')  # pose model
weapon_model = YOLO('yolov8s_pistol.pt')  # weapon detection model

# Classifications in models
pose_names = pose_model.names
weapon_names = weapon_model.names

# IDs of interest for the models
human_ids = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 34, 42, 43, 76]
weapon_ids = [0,1]

# Helper function to find bottom center of a detected object
def botm_center(x1, y1, x2, y2):
    x = int((x1 + x2) / 2)
    y = int(max(y1, y2))
    return [x, y]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Object detection for weapons
    weapon_results = weapon_model(frame, stream=True)
    weapon_centroids = []

    for result in weapon_results:
        boxes = result.boxes
        if boxes.shape[0] > 0:
            xyxy_array = boxes.xyxy.cpu().numpy()
            cls_array = boxes.cls.cpu().numpy()
            for xyxy, cls in zip(xyxy_array, cls_array):
                if int(cls) in weapon_ids:
                    x1, y1, x2, y2 = map(int, xyxy)
                    weapon_centroids.append(botm_center(x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Object detection for humans
    pose_results = pose_model.track(frame, stream=True)

    for result in pose_results:
        boxes = result.boxes
        if boxes.shape[0] > 0:
            if boxes.id is not None:
                ids_array = boxes.id.cpu().numpy()
            else:
                ids_array = None
            xyxy_array = boxes.xyxy.cpu().numpy()
            cls_array = boxes.cls.cpu().numpy()

            for xyxy, cls in zip(xyxy_array, cls_array):
                if int(cls) in human_ids:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check who is holding the weapon
            if weapon_centroids:
                keypoints = result.keypoints.xy.cpu().numpy()
                suspect_ids = []

                for center in weapon_centroids:
                    min_dist = 500
                    suspect_id = None
                    if ids_array is not None:
                        for id, keypoint in zip(ids_array, keypoints):
                            left_wrist = keypoint[9]
                            right_wrist = keypoint[10]
                            lor_dist = min(dist(left_wrist, center), dist(right_wrist, center))

                            if lor_dist < min_dist:
                                min_dist = lor_dist
                                suspect_id = int(id)
                    else:
                        ids_index = -1
                        for keypoint in keypoints:
                            ids_index += 1
                            left_wrist = keypoint[9]
                            right_wrist = keypoint[10]
                            lor_dist = min(dist(left_wrist, center), dist(right_wrist, center))

                            if lor_dist < min_dist:
                                min_dist = lor_dist
                                suspect_id = ids_index

                    if suspect_id not in suspect_ids and suspect_id is not None:
                        suspect_ids.append(suspect_id)

                if suspect_ids:
                    for id in suspect_ids:
                        x1, y1, x2, y2 = map(int, xyxy_array[id-1])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Suspect: {suspect_ids.index(id)+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    cv2.imshow('camera live feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
