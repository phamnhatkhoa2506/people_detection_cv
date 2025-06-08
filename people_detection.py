import os
import sys
import numpy as np
import cv2
import imutils
import torch
import supervision as sv
from typing import Optional
from ultralytics import YOLO
from datetime import datetime
from PIL import Image


# Configuration parameters
WIDTH = 800
start_status = "Loading..."
people_status = "People"
not_people_status = "Not people"

cls0_rect_color = (255, 255, 255)    # Yellow for not people (BGR)
cls1_rect_color = (255, 255, 0)      # Green for people (BGR)
conf_color = (255, 255, 0)           # Cyan for confidence score (BGR)
status_color = (0, 0, 255)           # Red for status (BGR)
header_color = (0, 255, 255)         # Yellow for header (BGR)
footer_color = (0, 255, 255)         # Yellow for footer (BGR)

frame_name = "People Detection"
quit_key = 'q'


class PeopleDetector:
    """Class to handle people detection using YOLO model."""

    def __init__(
        self,
        weights_path: str,
        input_path: str,
        output_path: str = ""
    ) -> None:
        
        self.input_path = input_path
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weights_path)
        self.cap = self._initialize_video_capture()
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0

        print(f"[INFO] Using device: {self.device}")

    def _load_model(self, weights_path: str) -> YOLO:
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")
            
            model = YOLO(weights_path).to(self.device)

            if not hasattr(model, 'predict'):
                raise AttributeError("Loaded model has no predict method")
            
            print(f"[INFO] Successfully loaded model from {weights_path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            sys.exit(1)

    def _initialize_video_capture(self) -> cv2.VideoCapture:
        if not os.path.exists(self.input_path):
            print(f"[ERROR] Input video not found at: {self.input_path}")
            sys.exit(1)

        cap = cv2.VideoCapture(self.input_path)

        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {self.input_path}")
            sys.exit(1)

        return cap
        
    def _draw_detection(
        self,
        frame: np.ndarray,
        box_data: np.ndarray
    ) -> str:
        xywh =  np.array(box_data.boxes.xywh.cpu()).astype("int32")
        xyxy =  np.array(box_data.boxes.xyxy.cpu()).astype("int32")
        cc_data = np.array(box_data.boxes.data.cpu())
        status = start_status
        
        for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, class_) in zip(xyxy, xywh, cc_data):
            if class_ == 1:
                self._draw_people_box(frame, x1, y1, w, h, conf)
                status = people_status
            elif class_ == 0 and conf > 0.8:
                self._draw_normal_box(frame, x1, y1, w, h, conf)
                status = not_people_status

        return status
    
    def _draw_people_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        w: int,
        h: int,
        conf: float
    ) -> None:
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), cls1_rect_color, 2)
        
        center_x = int(x1 + w / 2)
        cv2.circle(frame, (center_x, y1), 6, (0, 0, 255), -1)

        conf_text = f"{np.round(conf * 100, 2)}%"
        cv2.putText(frame, conf_text, (x1 + 10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    def _draw_normal_box(
        self, 
        frame: np.ndarray, 
        x1: int, 
        y1: int, 
        w: int, 
        h: int, 
        conf: float
    ) -> None:
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), cls0_rect_color, 2)
        conf_text = f"{np.round(conf * 100, 2)}%"
        cv2.putText(frame, conf_text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    def _setup_video_writer(self, frame: np.ndarray) -> None:
        if self.output_path and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    def _draw_header_footer(self, frame: np.ndarray, status: str) -> None:
        # Header
        header_text = f"Shoplifting Detection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(frame, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2)
        
        # Footer
        footer_text = f"Frame: {self.frame_count} | Status: {status}"
        frame_height = frame.shape[0]
        cv2.putText(frame, footer_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, footer_color, 2)

    def process_video(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] End of video reached")
                break

            self.frame_count += 1
            frame = imutils.resize(frame, width=WIDTH)

            try:
                result = self.model.predict(frame)
                if result is None or not result:
                    print(f"[WARNING] Prediction empty for frame {self.frame_count}")
                    continue
            except Exception as e:
                print(f"[ERROR] Prediction failed at frame {self.frame_count}: {str(e)}")
                break

            status = start_status
            if len(result[0].boxes) > 0:
                status = self._draw_detection(frame, result[0])
            
            # Draw header and footer
            self._draw_header_footer(frame, status)
            
            # Status text (kept for backward compatibility)
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, status_color, 2)
            cv2.imshow(frame_name, frame)

            self._setup_video_writer(frame)
            if self.writer:
                self.writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                print("[INFO] Quit key pressed")
                break

    def cleanup(self) -> None:
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated successfully")

    def detect_image(self, input_path: str, output_path: str) -> None:
        image = cv2.imread(input_path)

        model = YOLO('yolov8n.pt')

        results = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        Image.fromarray(annotated_image).save(output_path)

        
def main():
    # Specify an output path to save the video
    output_video_path = "outputs/people_output_1.avi"
    
    detector = PeopleDetector(
        weights_path="./runs/detect/train/weights/best.pt",
        input_path="./inputs/demo1.mp4",
        output_path=output_video_path  # Provide the output path here
    )
    detector.process_video()
    detector.cleanup()

    print(f"[INFO] Video saved to {output_video_path}")


if __name__ == '__main__':
    main()