import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from scipy.spatial import distance
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseState(Enum):
    DOWN = 0
    GOING_UP = 1
    UP = 2
    GOING_DOWN = 3

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,  # Lowered for better detection
            min_tracking_confidence=0.3    # Lowered for better detection
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def analyze_frame(self, frame):
        """Process a frame and return pose landmarks"""
        try:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return results.pose_landmarks
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
            return None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate the angle between three points"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        except:
            return 0
    
    @staticmethod
    def calculate_torso_angle(shoulder, hip, knee):
        """Calculate torso angle relative to horizontal"""
        try:
            # Vector from hip to shoulder (torso)
            torso_vector = (shoulder[0] - hip[0], shoulder[1] - hip[1])
            
            # Calculate angle between torso and horizontal
            horizontal_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
            horizontal_angle = abs(horizontal_angle)  # Make it positive
            
            return horizontal_angle
        except:
            return 0
    
    @staticmethod
    def get_landmark_coordinates(landmarks, image_shape):
        """Extract coordinates of key landmarks"""
        h, w = image_shape[:2]
        keypoints = {}
        
        if landmarks:
            for idx, lm in enumerate(landmarks.landmark):
                if idx in [11, 12, 23, 24, 25, 26, 27, 28]:  # Shoulders, hips, knees, ankles
                    keypoints[idx] = (int(lm.x * w), int(lm.y * h))
        
        return keypoints

class SitUpAnalyzer:
    def __init__(self):
        self.state = PoseState.DOWN
        self.correct_count = 0
        self.incorrect_count = 0
        self.rep_in_progress = False
        self.max_torso_angle = 0
        self.min_torso_angle = 180
        self.rep_data = {
            'max_angle': 0,
            'min_angle': 180,
            'hip_movement': 0,
            'knee_movement': 0
        }
        
        # Thresholds (adjustable)
        self.UP_ANGLE_THRESHOLD = 60  # Lowered for better detection
        self.DOWN_ANGLE_THRESHOLD = 40  # Adjusted
        self.CORRECT_ANGLE_RANGE = (60, 100)  # Wider range
        self.HIP_MOVEMENT_THRESHOLD = 0.15  # Increased threshold
        self.KNEE_MOVEMENT_THRESHOLD = 0.08  # Increased threshold
        
        # For debug
        self.last_feedback = "Starting..."
        self.missing_keypoints = []
        
    def analyze_situp(self, keypoints, frame_shape):
        """Analyze current frame for sit-up form and count"""
        if not keypoints or len(keypoints) < 4:  # Reduced minimum keypoints
            self.missing_keypoints = []
            for idx in [11, 12, 23, 24, 25, 26]:
                if idx not in keypoints:
                    self.missing_keypoints.append(idx)
            return f"Missing keypoints: {self.missing_keypoints}", False, 0
        
        # Get key points (using available points)
        left_shoulder = keypoints.get(11, (0, 0))
        right_shoulder = keypoints.get(12, (0, 0))
        left_hip = keypoints.get(23, (0, 0))
        right_hip = keypoints.get(24, (0, 0))
        left_knee = keypoints.get(25, (0, 0))
        right_knee = keypoints.get(26, (0, 0))
        
        # Calculate midpoints
        shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                   (left_shoulder[1] + right_shoulder[1]) // 2)
        hip = ((left_hip[0] + right_hip[0]) // 2, 
              (left_hip[1] + right_hip[1]) // 2)
        knee = ((left_knee[0] + right_knee[0]) // 2, 
               (left_knee[1] + right_knee[1]) // 2)
        
        # Calculate torso angle relative to horizontal
        torso_angle = PoseAnalyzer.calculate_torso_angle(shoulder, hip, knee)
        
        # Track min/max angles during repetition
        if self.rep_in_progress:
            self.rep_data['max_angle'] = max(self.rep_data['max_angle'], torso_angle)
            self.rep_data['min_angle'] = min(self.rep_data['min_angle'], torso_angle)
            
            # Track hip movement (vertical displacement normalized by frame height)
            if hasattr(self, 'initial_hip_y'):
                hip_movement = abs(hip[1] - self.initial_hip_y) / frame_shape[0]
                self.rep_data['hip_movement'] = max(self.rep_data['hip_movement'], hip_movement)
            
            # Track knee movement (change in position from start)
            if hasattr(self, 'initial_knee'):
                knee_movement = distance.euclidean(knee, self.initial_knee) / frame_shape[0]
                self.rep_data['knee_movement'] = max(self.rep_data['knee_movement'], knee_movement)
        
        # State machine for rep counting
        feedback = "Good form"
        rep_completed = False
        
        if self.state == PoseState.DOWN:
            if torso_angle > self.UP_ANGLE_THRESHOLD:
                self.state = PoseState.GOING_UP
                self.rep_in_progress = True
                self.initial_hip_y = hip[1]
                self.initial_knee = knee
                self.rep_data = {
                    'max_angle': torso_angle,
                    'min_angle': torso_angle,
                    'hip_movement': 0,
                    'knee_movement': 0
                }
                feedback = "Going up!"
                
        elif self.state == PoseState.GOING_UP:
            if torso_angle >= self.rep_data['max_angle']:
                self.rep_data['max_angle'] = torso_angle
                
            if torso_angle > self.UP_ANGLE_THRESHOLD + 5:  # Smaller buffer
                self.state = PoseState.UP
                feedback = "Up position!"
                
        elif self.state == PoseState.UP:
            if torso_angle < self.UP_ANGLE_THRESHOLD:
                self.state = PoseState.GOING_DOWN
                feedback = "Going down!"
                
        elif self.state == PoseState.GOING_DOWN:
            if torso_angle < self.DOWN_ANGLE_THRESHOLD:
                self.state = PoseState.DOWN
                rep_completed = True
                self.rep_in_progress = False
                
                # Check if rep was correct
                is_correct = self._check_form_correctness()
                if is_correct:
                    self.correct_count += 1
                    feedback = "Correct rep! +1"
                else:
                    self.incorrect_count += 1
                    feedback = "Incorrect form"
        
        self.last_feedback = feedback
        return feedback, rep_completed, torso_angle
    
    def _check_form_correctness(self):
        """Check if the completed repetition had correct form"""
        # Check if reached proper up position
        if not (self.CORRECT_ANGLE_RANGE[0] <= self.rep_data['max_angle'] <= self.CORRECT_ANGLE_RANGE[1]):
            return False
        
        # Check if returned to proper down position
        if self.rep_data['min_angle'] > self.DOWN_ANGLE_THRESHOLD + 15:
            return False
        
        # Check for excessive hip movement (cheating by lifting hips)
        if self.rep_data['hip_movement'] > self.HIP_MOVEMENT_THRESHOLD:
            return False
        
        # Check for excessive knee movement (swinging legs)
        if self.rep_data['knee_movement'] > self.KNEE_MOVEMENT_THRESHOLD:
            return False
            
        return True
    
    def get_counts(self):
        """Return current counts"""
        total = self.correct_count + self.incorrect_count
        accuracy = self.correct_count / total * 100 if total > 0 else 0
        
        return {
            'correct': self.correct_count,
            'incorrect': self.incorrect_count,
            'total': total,
            'accuracy': round(accuracy, 2)
        }
    
    def get_debug_info(self):
        """Get debug information"""
        return {
            'missing_keypoints': self.missing_keypoints,
            'state': self.state.name,
            'rep_in_progress': self.rep_in_progress,
            'last_feedback': self.last_feedback
        }