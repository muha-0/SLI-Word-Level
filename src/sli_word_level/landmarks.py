"""
MediaPipe-based landmark extraction utilities for ESL sign sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


@dataclass
class LandmarkExtractor:
    """
    Wrapper around MediaPipe Hands and Pose for landmark extraction.

    Uses static_image_mode=True since we treat each frame independently
    in the real-time loop.
    """

    def __post_init__(self) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
        )
        self.pose = mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5
        )

    def extract(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Extract left-hand, pose, and right-hand landmarks from a single BGR frame.

        Returns:
            landmarks: (75, 3) array stacked as [left hand (21); pose (33); right hand (21)]
            hand_count: number of detected hands (0, 1, or 2)
            pose_detected: 1 if pose landmarks detected, else 0
        """
        # Default to zeros if something is missing
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((33, 3), dtype=np.float32)

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(image_rgb)
        pose_results = self.pose.process(image_rgb)

        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                if hand_label == "Left":
                    left_hand = coords
                else:
                    right_hand = coords

        if pose_results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                dtype=np.float32,
            )

        all_landmarks = np.vstack([left_hand, pose, right_hand])
        hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        pose_detected = 1 if pose_results.pose_landmarks else 0

        return all_landmarks, hand_count, pose_detected


def split_multistream(sequence: np.ndarray):
    """
    Split a stacked landmark sequence into (pose, left_hand, right_hand).

    Args:
        sequence: (T, 75, 3) array where 75 = 21 (left hand) + 33 (pose) + 21 (right hand)

    Returns:
        pose: (T, 33, 3)
        left: (T, 21, 3)
        right: (T, 21, 3)
    """
    # left hand: 0..20, pose: 21..53, right hand: 54..74
    pose = sequence[:, 21:54, :]
    left = sequence[:, :21, :]
    right = sequence[:, 54:, :]
    return pose, left, right
