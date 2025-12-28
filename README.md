# Facial Emotion Detection and Eye Tracking

A real-time AI-powered system for detecting facial emotions and tracking eyes using **InsightFace** and **HSEmotion**. Optimized for performance and stability on macOS.

## Features
- **Accurate Face Detection**: Powered by InsightFace's `buffalo_l` model.
- **106-Point Landmarks**: Detailed facial feature extraction.
- **Emotion Recognition**: Efficient and stable ONNX-based emotion detection using `hsemotion-onnx`.
- **Eye Tracking**: Real-time eye center localization.
- **Lag Optimization**: Intelligent inference skipping and result caching for high FPS.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

- Press **'q'** to exit the video feed.

## Dependencies
- `insightface`
- `hsemotion-onnx`
- `onnxruntime`
- `opencv-python`
- `numpy`

## Note
On the first run, the application will download necessary pre-trained models (~500MB). Ensure you have sufficient disk space.
