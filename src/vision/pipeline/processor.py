from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

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

