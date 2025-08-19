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
        self._heatmap_annotator = sv.HeatMapAnnotator()
        self._blur_annotator = sv.BlurAnnotator()

        # Zone components (initialized in process_video when resolution is known)
        self._zone = None
        self._zone_annotator = None
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

        # Define a polygonal zone using normalized coordinates, then scale to pixels
        width, height = video_info.width, video_info.height
        polygon_norm = np.array(
            [
                [0.25, 0.70],
                [0.75, 0.70],
                [0.85, 0.95],
                [0.15, 0.95],
            ],
            dtype=np.float32,
        )
        polygon_px = np.column_stack((polygon_norm[:, 0] * width, polygon_norm[:, 1] * height)).astype(
            np.int32
        )
        self._zone = sv.PolygonZone(polygon=polygon_px, frame_resolution_wh=(width, height))
        self._zone_annotator = sv.PolygonZoneAnnotator(zone=self._zone, color=sv.Color.blue())

        # Use a broadly supported codec for MP4 writing in headless envs
        with sv.VideoSink(output_path, video_info, codec="mp4v") as sink:
            for frame in frames:
                # Inference
                result = self._model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = self._filter_detections(detections)

                # Tracking
                tracked = self._tracker.update_with_detections(detections)

                # Start from base frame and apply GDPR-compliant blurring for persons
                annotated = frame.copy()
                class_ids = tracked.class_id if tracked.class_id is not None else np.full(len(tracked), -1)
                person_mask = class_ids == 0
                if np.any(person_mask):
                    det_person = tracked[person_mask]
                    annotated = self._blur_annotator.annotate(scene=annotated, detections=det_person)

                # Heatmap next for path visualization (over blurred content)
                annotated = self._heatmap_annotator.annotate(scene=annotated, detections=tracked)

                # Geofencing and dwell-time count
                if self._zone_annotator is not None and self._zone is not None:
                    self._zone_annotator.annotate(annotated)
                    in_zone_mask = self._zone.trigger(tracked)
                    dwell_count = int(np.count_nonzero(in_zone_mask))
                    cv2.putText(
                        annotated,
                        f"In Zone: {dwell_count}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Near-miss detection between persons and vehicles (cars/trucks)
                boxes = tracked.xyxy.astype(np.float32)
                centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
                vehicle_mask = np.isin(class_ids, np.array([2, 7]))
                near_mask = np.zeros(len(tracked), dtype=bool)
                threshold_px = 75.0
                person_idx = np.where(person_mask)[0]
                vehicle_idx = np.where(vehicle_mask)[0]
                if len(person_idx) > 0 and len(vehicle_idx) > 0:
                    for pi in person_idx:
                        pc = centers[pi]
                        vc = centers[vehicle_idx]
                        dists = np.linalg.norm(vc - pc, axis=1)
                        if dists.size:
                            min_j = int(np.argmin(dists))
                            if dists[min_j] < threshold_px:
                                near_mask[pi] = True
                                near_mask[vehicle_idx[min_j]] = True

                # Labels with near tag
                labels = []
                for i in range(len(tracked)):
                    class_id = int(class_ids[i])
                    confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    class_name = self._class_names.get(class_id, str(class_id))
                    tag = " NEAR" if near_mask[i] else ""
                    labels.append(f"{class_name} #{track_id}{tag} {confidence:.2f}")

                # Annotate with different colors for near-miss
                normal_mask = ~near_mask
                if np.any(normal_mask):
                    det_norm = tracked[normal_mask]
                    labels_norm = [lbl for lbl, m in zip(labels, normal_mask) if m]
                    annotated = self._box_annotator.annotate(scene=annotated, detections=det_norm)
                    annotated = self._label_annotator.annotate(
                        scene=annotated, detections=det_norm, labels=labels_norm
                    )
                if np.any(near_mask):
                    det_near = tracked[near_mask]
                    labels_near = [lbl for lbl, m in zip(labels, near_mask) if m]
                    # Manually draw near-miss boxes and labels in red
                    boxes_near = det_near.xyxy.astype(int)
                    for (x1, y1, x2, y2), label in zip(boxes_near, labels_near):
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                sink.write_frame(annotated)

