# People Detection System

A real-time people detection system built using YOLOv8 and OpenCV. This project can detect and track people in video streams and images, with the ability to distinguish between people and non-people objects.

## Features

- Real-time people detection in video streams
- Support for both video and image processing
- Confidence score display for each detection
- Customizable bounding box colors for different classes
- Video output saving capability
- Frame counter and timestamp display
- GPU acceleration support (when available)

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Supervision
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd people_detection_cv
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
people_detection_cv/
├── inputs/              # Input videos and images
├── outputs/            # Processed output videos and images
├── runs/              # Training runs and model weights
├── people_detection.py # Main detection script
├── train.py           # Training script
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Usage

### Video Detection

To process a video file:

```python
from people_detection import PeopleDetector

detector = PeopleDetector(
    weights_path="./runs/detect/train/weights/best.pt",
    input_path="./inputs/your_video.mp4",
    output_path="./outputs/output_video.avi"
)
detector.process_video()
detector.cleanup()
```

### Image Detection

To process a single image:

```python
from people_detection import PeopleDetector

detector = PeopleDetector(
    weights_path="./runs/detect/train/weights/best.pt",
    input_path="",  # Not needed for image detection
    output_path=""  # Not needed for image detection
)
detector.detect_image(
    input_path="./inputs/your_image.jpg",
    output_path="./outputs/output_image.jpg"
)
```

## Configuration

The system uses the following default parameters that can be modified in the code:

- Frame width: 800 pixels
- Bounding box colors:
  - People: Green (255, 255, 0)
  - Non-people: White (255, 255, 255)
  - Confidence score: Cyan (255, 255, 0)
  - Status text: Red (0, 0, 255)
  - Header/Footer: Yellow (0, 255, 255)

## Output

The system provides:
- Real-time visualization with bounding boxes
- Confidence scores for each detection
- Timestamp and frame counter
- Status indicator showing detection results
- Saved output video/image with annotations

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 