import cv2
import numpy as np

class SORTTracker:
    def __init__(self):
        # Initialize variables to store tracks
        self.tracks = []
        self.track_id_counter = 1

        # Define parameters for SORT
        self.iou_threshold = 0.5
        self.max_age = 5

    def update(self, detections):
        # Predict and update each track
        for track in self.tracks:
            track.predict()

        # Associate detections with tracks
        matched_tracks = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))

        for track in self.tracks:
            best_detection_idx = track.match_detection(detections, self.iou_threshold)
            if best_detection_idx is not None:
                track.update(detections[best_detection_idx])
                matched_tracks.append(track)
                unmatched_detections.remove(best_detection_idx)
            else:
                unmatched_tracks.append(track)

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = Track(self.track_id_counter)
            new_track.initiate(detections[detection_idx])
            self.track_id_counter += 1
            self.tracks.append(new_track)

        # Increment age of unmatched tracks
        for track in unmatched_tracks:
            track.increment_age()

        # Remove old tracks
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

    def draw_tracks(self, frame):
        # Draw bounding boxes and track IDs on the frame
        for track in self.tracks:
            track.draw(frame)


class Track:
    def __init__(self, track_id):
        self.track_id = track_id
        self.bbox = None
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.max_age = 5

        # Define Kalman filter parameters
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.01
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                 [0, 1]], np.float32) * 0.1

    def d(self, detection):
        # Initialize track with the first measurement
        self.bbox = detection
        self.kf.statePre = np.array(detection, np.float32).reshape(4, 1)
        self.age = 1

    def predict(self):
        # Predict the next state of the track
        self.bbox = self.kf.predict()
        self.age += 1

    def update(self, detection):
        # Update the track with a new measurement
        self.kf.correct(np.array(detection, np.float32).reshape(2, 1))
        self.age = 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0

    def match_detection(self, detections, iou_threshold):
        # Match the track with a detection based on IoU
        best_detection_idx = None
        max_iou = -np.inf

        for idx, detection in enumerate(detections):
            iou = self.compute_iou(detection)
            if iou > max_iou and iou > iou_threshold:
                max_iou = iou
                best_detection_idx = idx

        return best_detection_idx

    def compute_iou(self, detection):
        # Compute IoU (Intersection over Union) with a detection
        x_tl_track, y_tl_track, w_track, h_track = self.bbox
        x_tl_det, y_tl_det, w_det, h_det = detection

        x_br_track, y_br_track = x_tl_track + w_track, y_tl_track + h_track
        x_br_det, y_br_det = x_tl_det + w_det, y_tl_det + h_det

        x_tl_intersection = max(x_tl_track, x_tl_det)
        y_tl_intersection = max(y_tl_track, y_tl_det)
        x_br_intersection = min(x_br_track, x_br_det)
        y_br_intersection = min(y_br_track, y_br_det)

        intersection_area = max(0, x_br_intersection - x_tl_intersection + 1) * max(0, y_br_intersection - y_tl_intersection + 1)

        area_track = w_track * h_track
        area_det = w_det * h_det

        iou = intersection_area / float(area_track + area_det - intersection_area)

        return iou

    def increment_age(self):
        # Increment age of the track
        self.age += 1
        self.consecutive_invisible_count += 1

    def is_deleted(self):
        # Check if the track is deleted based on age
        return self.consecutive_invisible_count > self.max_age

    def draw(self, frame):
        # Draw bounding box and track ID on the frame
        x, y, w, h = map(int, self.bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {self.track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Example usage
if __name__ == "__main__":
    tracker = SORTTracker()
    
    # Load pre-trained model and its configuration
    model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")  # Can use other models like 'ssd_mobilenet_v3_large_coco.pbtxt' and 'frozen_inference_graph.pb'
    # Load class labels
    with open("coco.names.txt", "r") as f:
        classes = f.read().strip().split("\n")

    cap = cv2.VideoCapture('demo.mp4')  # Use 0 for default webcam

    while True:
        ret, frame = cap.read()

        (H, W) = frame.shape[:2]

        # Preprocess frame for the model
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Pass the blob through the network and get detections
        model.setInput(blob)
        layer_names = model.getLayerNames()
        # import pdb;pdb.set_trace()
        output_layers = []
        for i in model.getUnconnectedOutLayers():
            # print(i)
            out = layer_names[i - 1]
            output_layers.append(out)
        # output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        outputs = model.forward(output_layers)

        # Process detections
        bounding_boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Minimum confidence threshold
                    # Scale bounding box coordinates to the original image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # bounding_boxes.append([centerX, centerY, width, height])
                    # Calculate top-left corner coordinates
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    bounding_boxes.append([x, y, x+width, y+height])
                    # Draw bounding box and label on the image
                    color = (0, 255, 0)  # BGR format
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    text = "{}: {:.4f}".format(classes[class_id], confidence)
                    # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections = bounding_boxes

        # Example detections (bounding boxes)
        # detections = [[100, 100, 50, 50], [200, 200, 60, 60], [300, 300, 70, 70]]

        tracker.update(detections)
        tracker.draw_tracks(frame)

        cv2.imshow("SORT Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
