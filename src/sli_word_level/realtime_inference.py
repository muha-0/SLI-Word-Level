"""
Real-time ESL word-level inference using a multistream LSTM model.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from .config import (
    MODEL_PATH,
    WINDOW_SIZE,
    MIN_FRAMES,
    CONFIDENCE_THRESHOLD,
    LABEL_TO_FOLDER,
)
from .landmarks import LandmarkExtractor, split_multistream
from .grammar_correction import correct_text


class RealTimeInterpreter:
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)
        self.extractor = LandmarkExtractor()
        self.sliding_window: List[np.ndarray] = []
        self.collecting: bool = False

    def _pad_or_sample_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Ensure we have exactly WINDOW_SIZE frames by sampling or padding.
        """
        pad_frame = np.zeros((75, 3), dtype=np.float32)

        if len(frames) < WINDOW_SIZE:
            frames = frames + [pad_frame] * (WINDOW_SIZE - len(frames))
        else:
            # Uniform sampling down to WINDOW_SIZE
            step = max(1, len(frames) // WINDOW_SIZE)
            sampled = []
            for i in range(0, len(frames), step):
                if len(sampled) < WINDOW_SIZE:
                    sampled.append(frames[i])
                else:
                    break
            while len(sampled) < WINDOW_SIZE:
                sampled.append(pad_frame)
            frames = sampled

        return np.asarray(frames, dtype=np.float32)

    def predict_frame(self, frame_bgr: np.ndarray) -> str:
        """
        Process a single frame and return a predicted word if a sign has finished.
        Otherwise returns an empty string.
        """
        landmarks, hand_count, _ = self.extractor.extract(frame_bgr)

        # Hands visible: keep collecting frames
        if hand_count > 0:
            self.collecting = True
            self.sliding_window.append(landmarks)
            return ""

        # No hands visible: if we were collecting, finalize a prediction
        if self.collecting:
            self.collecting = False
            total = len(self.sliding_window)

            if total < MIN_FRAMES:
                self.sliding_window.clear()
                return ""

            sequence = self._pad_or_sample_frames(self.sliding_window)
            self.sliding_window.clear()

            pose, left, right = split_multistream(sequence)

            preds = self.model.predict(
                [pose[np.newaxis, :], left[np.newaxis, :], right[np.newaxis, :]],
                verbose=0,
            )
            predicted_class = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds))

            print(f"[SLI] Confidence: {confidence:.2f}")
            if confidence >= CONFIDENCE_THRESHOLD:
                word = LABEL_TO_FOLDER.get(predicted_class, "")
                print(f"[SLI] Prediction: {word}")
                return word

        return ""


def run_webcam_demo(use_grammar_correction: bool = False) -> None:
    """
    Launch a webcam window and run real-time sign language interpretation.
    Press 'q' to quit.
    """
    interpreter = RealTimeInterpreter()
    cap = cv2.VideoCapture(0)

    sentence: str = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        word = interpreter.predict_frame(frame)

        if word:
            sentence = f"{sentence}{word} "

            if use_grammar_correction:
                sentence = correct_text(sentence)

        cv2.putText(
            frame,
            sentence,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("ESL Word-Level Interpreter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo(use_grammar_correction=True)
