import cv2
import numpy as np
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionGazeTracker:
import threading
import time

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class EmotionGazeTracker:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
        
        # State & Smoothing
        self.prev_faces = []
        self.frame_count = 0
        self.skip_frames = 2
        self.ema_landmarks = {} # For multiple faces, used as cache
        self.alpha = 0.4 # EMA parameter (0-1), lower is smoother but slower
        
    def process_frame(self, frame):
        self.frame_count += 1
        
        # Inference frame
        if self.frame_count % (self.skip_frames + 1) == 0 or not self.prev_faces:
            faces = self.app.get(frame)
            for face in faces:
                try:
                    bbox = face.bbox.astype(int)
                    face_roi = frame[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
                    if face_roi.size > 0:
                        emotion, scores = self.fer.predict_emotions(face_roi)
                        face['emotion'] = emotion
                except Exception:
                    face['emotion'] = "..."
            self.prev_faces = faces

        # Smoothing and Drawing
        for i, face in enumerate(self.prev_faces):
            bbox = face.bbox.astype(int)
            landmarks = face.landmark_2d_106
            
            if landmarks is not None:
                # Apply EMA Smoothing to Landmarks
                if i not in self.ema_landmarks:
                    self.ema_landmarks[i] = landmarks
                else:
                    self.ema_landmarks[i] = self.alpha * landmarks + (1 - self.alpha) * self.ema_landmarks[i]
                
                # Draw Smoothed Landmarks
                curr_landmarks = self.ema_landmarks[i].astype(int)
                for pt in curr_landmarks:
                    cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                
                # Eye Tracking on smoothed landmarks
                left_eye_indices = [35, 41, 42, 39, 37, 36]
                right_eye_indices = [89, 95, 96, 93, 91, 90]
                left_center = np.mean(curr_landmarks[left_eye_indices], axis=0).astype(int)
                right_center = np.mean(curr_landmarks[right_eye_indices], axis=0).astype(int)
                cv2.circle(frame, tuple(left_center), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, tuple(right_center), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # Draw Bounding Box and Emotion
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1, lineType=cv2.LINE_AA)
            emotion = face.get('emotion', "")
            cv2.putText(frame, f"{emotion}", (bbox[0], bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, lineType=cv2.LINE_AA)

        return frame

import argparse

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion and Eye Tracking")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    tracker = EmotionGazeTracker()
    vs = VideoStream(src=args.camera).start()
    
    # Wait for camera to warm up
    time.sleep(1.0)

    print(f"Starting optimized stream from camera {args.camera}. Press 'q' to quit.")
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
        
        # Work on a copy to keep original clean if needed
        display_frame = frame.copy()
        
        # Process frame
        processed_frame = tracker.process_frame(display_frame)
        
        cv2.imshow('Emotion and Eye Tracking (Smoothed)', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
