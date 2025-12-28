import cv2
import numpy as np
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionGazeTracker:
    def __init__(self):
        # Initialize InsightFace for detection and 106-point landmarks
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize HSEmotion for stable, fast emotion recognition
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
        
        # Performance variables
        self.prev_faces = []
        self.frame_count = 0
        self.skip_frames = 2 # Process every 3rd frame
        
    def process_frame(self, frame):
        self.frame_count += 1
        
        # Run heavy inference only every skip_frames
        if self.frame_count % (self.skip_frames + 1) == 0 or not self.prev_faces:
            faces = self.app.get(frame)
            # Pre-calculate emotions for all faces in this inference frame
            for face in faces:
                try:
                    bbox = face.bbox.astype(int)
                    face_roi = frame[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
                    if face_roi.size > 0:
                        emotion, scores = self.fer.predict_emotions(face_roi)
                        face['emotion'] = emotion
                except Exception:
                    face['emotion'] = "Detecting..."
            self.prev_faces = faces
        
        # Draw results (either new or cached)
        for face in self.prev_faces:
            # 1. Draw Bounding Box
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 2. Extract Landmarks (106 keypoints)
            landmarks = face.landmark_2d_106
            if landmarks is not None:
                for pt in landmarks.astype(int):
                    cv2.circle(frame, tuple(pt), 1, (255, 255, 0), -1)
            
            # 3. Emotion Display (using cached result)
            emotion = face.get('emotion', "Analyzing...")
            cv2.putText(frame, f"Emotion: {emotion}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 4. Eye Tracking
            if landmarks is not None:
                left_eye_indices = [35, 41, 42, 39, 37, 36]
                right_eye_indices = [89, 95, 96, 93, 91, 90]
                
                def get_eye_center(indices):
                    pts = landmarks[indices]
                    return np.mean(pts, axis=0).astype(int)

                left_center = get_eye_center(left_eye_indices)
                right_center = get_eye_center(right_eye_indices)
                cv2.circle(frame, tuple(left_center), 2, (0, 0, 255), -1)
                cv2.circle(frame, tuple(right_center), 2, (0, 0, 255), -1)

        return frame

def main():
    tracker = EmotionGazeTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    print("Starting video feed. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = tracker.process_frame(frame)
        
        # Display the resulting frame
        cv2.imshow('Emotion and Eye Tracking', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
