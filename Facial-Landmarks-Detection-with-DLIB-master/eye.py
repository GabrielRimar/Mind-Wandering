import numpy as np
import cv2 
import math

from pupil import Pupil

class Eye:
    def __init__(self, original_frame, landmarks):
        # right_eye : bool - will determine right eye or left 
        self.landmark_points = None 
        self.frame = None
        self.center = None
        self.pupil = None
        self.ear = None
        self.origin = None

        if original_frame is not None and landmarks is not None:
            self._analyze(original_frame, landmarks)
        else:
            raise ValueError("no right_eye, original_frame or landmarks detected")
            
    
    def _isolate(self, frame, landmarks_points):
        """
            Method isolates the eye.
        """
        region = landmarks_points
        region = region.astype(np.int32)

        height, width = frame.shape[:2] # dimantions of the frame
        black_frame = np.zeros((height, width), np.uint8) # array in the frame dimantions but all blacked filled with 0
        # np.uint8 used in 0 - 255 values 
        mask = mask = np.full((height, width), 255, np.uint8) # array filled with white values 255
        cv2.fillPoly(mask, [region], (0, 0, 0)) # poligon on the mask with a black cutout in the region of the eye
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask) # blacks out the entire face exept the eye

        # Cropping the eye
        margin = 5 # five pixel ofset
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = [min_x,min_y]

        
    def _analyze(self, original_frame, landmarks):
        #method will determine what eye and will isolate the frame

        self.landmark_points = landmarks
        self._isolate(original_frame, landmarks)
        self.ear = self._calculate_EAR(self.landmark_points)
        self.center = self._eye_center(self.landmark_points)


        self.pupil = Pupil(self.frame)
    
    def pupils_detected(self):
        return (self.pupil.x is not None
            and self.pupil.y is not None
            and self.pupil.x is not None
            and self.pupil.y is not None)
    @staticmethod
    def _calculate_EAR(landmark_points):
        if len(landmark_points) != 6:
            raise ValueError(f"Expected 6 eye landmarks, got {len(landmark_points)}.")
        
        p = landmark_points

        a = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
        b = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
        c = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
        
        ear = (a + b)/ (2*c)
        return ear
    
    @staticmethod
    def _eye_center(landmarks):
        # center will be (0,0) for me and of the pupil located there the person is looking strate
        return landmarks.mean(axis=0)
