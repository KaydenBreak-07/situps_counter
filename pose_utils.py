import cv2
import mediapipe as mp
import numpy as np
from enum import Enum

class PoseState(Enum):
    UP = 1
    DOWN = 2
    UNKNOWN = 3

class PoseAnalyzer:
    def __init__(self, detection_conf=0.5, tracking_conf=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_keypoints(self, image):
        """Extract pose landmarks from image."""
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = {}
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                keypoints[idx] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
        return keypoints, results

    def draw_landmarks(self, image, results):
        """Draw pose landmarks on frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        return image

class SitUpAnalyzer:
    def __init__(self):
        self.state = PoseState.DOWN
        self.correct_count = 0
        self.incorrect_count = 0
        self.rep_in_progress = False
        self.rep_data = {
            'max_angle': 0,
            'min_angle': 180
        }
        self.last_feedback = "Starting..."
        self.missing_keypoints = []

        # Thresholds
        self.UP_ANGLE_THRESHOLD = 60
        self.DOWN_ANGLE_THRESHOLD = 40
        self.CORRECT_ANGLE_RANGE = (60, 100)

    def calculate_angle(self, a, b, c):
        """Calculate angle between 3 points (a-b-c)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return min(angle, 360 - angle)

    def analyze_situp(self, keypoints, frame_shape):
        """Process one frame and update sit-up counts."""
        h, w = frame_shape[:2]
        required = [11, 23, 25]  # shoulder, hip, knee
        self.missing_keypoints = [idx for idx in required if idx not in keypoints]

        if self.missing_keypoints:
            self.last_feedback = f"Missing keypoints: {self.missing_keypoints}"
            return None

        shoulder = (keypoints[11]['x'] * w, keypoints[11]['y'] * h)
        hip = (keypoints[23]['x'] * w, keypoints[23]['y'] * h)
        knee = (keypoints[25]['x'] * w, keypoints[25]['y'] * h)

        angle = self.calculate_angle(shoulder, hip, knee)

        # Track min/max for current rep
        self.rep_data['max_angle'] = max(self.rep_data['max_angle'], angle)
        self.rep_data['min_angle'] = min(self.rep_data['min_angle'], angle)

        # State machine
        if self.state == PoseState.DOWN and angle < self.UP_ANGLE_THRESHOLD:
            self.state = PoseState.UP
            self.rep_in_progress = True
            self.last_feedback = "Good! Now go back down."
        elif self.state == PoseState.UP and angle > self.DOWN_ANGLE_THRESHOLD:
            self.state = PoseState.DOWN
            if self.rep_in_progress:
                self._check_form_correctness()
                self.rep_in_progress = False

        return angle

    def _check_form_correctness(self):
        """Check whether the rep was correct or incorrect."""
        if self.rep_data['min_angle'] < self.CORRECT_ANGLE_RANGE[0] and \
           self.rep_data['max_angle'] > self.CORRECT_ANGLE_RANGE[1]:
            self.correct_count += 1
            self.last_feedback = "Correct sit-up!"
        else:
            self.incorrect_count += 1
            self.last_feedback = "Try to maintain proper form."

        # Reset rep data
        self.rep_data = {'max_angle': 0, 'min_angle': 180}

    def get_counts(self):
        return {
            "correct": self.correct_count,
            "incorrect": self.incorrect_count
        }

    def get_debug_info(self):
        return {
            "state": self.state.name,
            "rep_in_progress": self.rep_in_progress,
            "last_feedback": self.last_feedback,
            "missing_keypoints": self.missing_keypoints
        }
