from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
from loguru import logger

from vision.core.config import AppConfig


@dataclass
class InferenceResult:
    boxes: np.ndarray  # shape: (N, 4) in xyxy
    scores: np.ndarray  # shape: (N,)
    labels: list[str]


class FrameProcessor:
    def __init__(self, app_config: AppConfig) -> None:
        self.config = app_config
        logger.debug("Initialized FrameProcessor with config: {}", self.config.model_dump())

    def _open_source(self, source: str) -> cv2.VideoCapture:
        try:
            index = int(source)
            cap = cv2.VideoCapture(index)
        except ValueError:
            cap = cv2.VideoCapture(source)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.video.fps)

        if not cap.isOpened():
            msg = f"Unable to open video source: {source}"
            logger.error(msg)
            raise RuntimeError(msg)
        return cap

    def _frames(self, cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

    def _dummy_infer(self, frame: np.ndarray) -> InferenceResult:
        # Placeholder inference: draw a centered box
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = w // 4, h // 4
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        labels = ["object"]
        return InferenceResult(boxes=boxes, scores=scores, labels=labels)

    def _annotate(self, frame: np.ndarray, result: InferenceResult) -> np.ndarray:
        annotated = frame.copy()
        for (x1, y1, x2, y2), score, label in zip(
            result.boxes.astype(int), result.scores, result.labels
        ):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label}: {score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_stream(self, source: str, display: bool = False) -> None:
        cap = self._open_source(source)
        try:
            for frame in self._frames(cap):
                result = self._dummy_infer(frame)
                output = self._annotate(frame, result)
                if display:
                    cv2.imshow("YardVision", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                try:
                    cv2.destroyAllWindows()
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to destroy windows (likely headless environment)")


class VideoProcessor:
    """Batch video processor using YOLOv8 + ByteTrack via supervision.

    This processor reads frames from an input video, performs detection and
    tracking, annotates results, and writes an output video.
    """

    def __init__(self, model_path: str = "yolov8m.pt") -> None:
        # Lazy imports to avoid importing heavy deps where not needed
        from ultralytics import YOLO  # type: ignore[import-not-found]
        import supervision as sv  # type: ignore[import-not-found]

        self._sv = sv
        self._model = YOLO(model_path)
        # ByteTrack tracker
        self._tracker = sv.ByteTrack()
        # Annotators
        self._box_annotator = sv.BoundingBoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()
        # COCO class name mapping from model
        try:
            self._class_names = self._model.model.names  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            # Fallback to standard COCO mapping indices used by YOLOv8
            self._class_names = {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
            }

        logger.info("Initialized VideoProcessor with model: {}", model_path)

    def _filter_detections(self, detections: "np.ndarray | object") -> "object":
        """Filter detection classes to person (0), car (2), truck (7).

        Works with supervision.Detections instance which supports numpy-like
        indexing using a boolean mask.
        """
        sv = self._sv
        assert isinstance(detections, sv.Detections)
        allowed = np.array([0, 2, 7])
        mask = np.isin(detections.class_id, allowed)
        return detections[mask]

    def process_video(self, input_path: str, output_path: str) -> None:
        import supervision as sv  # type: ignore[import-not-found]

        video_info = sv.VideoInfo.from_video_path(input_path)
        frames = sv.get_video_frames_generator(input_path)

        # Use a broadly supported codec for MP4 writing in headless envs
        with sv.VideoSink(output_path, video_info, codec="mp4v") as sink:
            for frame in frames:
                # Inference
                result = self._model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = self._filter_detections(detections)

                # Tracking
                tracked = self._tracker.update_with_detections(detections)

                # Labels for annotation
                labels = []
                for i in range(len(tracked)):
                    class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                    confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    class_name = self._class_names.get(class_id, str(class_id))
                    labels.append(f"{class_name} #{track_id} {confidence:.2f}")

                # Annotation
                annotated = self._box_annotator.annotate(scene=frame.copy(), detections=tracked)
                annotated = self._label_annotator.annotate(
                    scene=annotated,
                    detections=tracked,
                    labels=labels,
                )

                sink.write_frame(annotated)

