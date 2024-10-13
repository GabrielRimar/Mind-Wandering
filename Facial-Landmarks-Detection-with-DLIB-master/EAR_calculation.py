"""
EAR Calculation
----------------
The Eye Aspect Ratio (EAR) is a metric used to estimate eye openness. It is commonly used in attention monitoring, drowsiness detection, and blink detection systems. EAR is calculated by taking the ratio of the vertical distance between the eyelids to the horizontal distance across the eye.

Formula:
--------
EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)

Where:
- P1, P2, P3, P4, P5, and P6 are the coordinates of the six facial landmarks around the eye:
    - P1: Outer corner of the eye
    - P2: Upper eyelid
    - P3: Lower eyelid
    - P4: Inner corner of the eye
    - P5: Lower eyelid
    - P6: Upper eyelid

Usage:
------
1. Extract facial landmarks using a facial landmark detector (e.g., Dlib or OpenCV).
2. Pass the 6 key points of each eye to the EAR function.
3. EAR values typically decrease when the eyes are closed (blinking), making it useful for blink detection.

Parameters:
-----------
eye: list
    A list of 6 (x, y) coordinates representing the eye's key landmarks.

Returns:
--------
float
    The Eye Aspect Ratio (EAR) value. Lower values (e.g., < 0.2) indicate that the eye is likely closed or blinking.
"""

import math
import os
import cv2
import dlib
from imutils import face_utils

class EARCalculator:

    def __init__(self, shape_predictor_path):
        if not cv2.os.path.isfile(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found at {shape_predictor_path}.")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
    
    @staticmethod
    def _distance(p1,p2):
        '''
        This method calculates the distance between two points, it's a static method.
        '''
        if p1 is None or p2 is None:
           
            raise ValueError("Point 1 or Point 2 is empty")

        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
        #hypot -> square root of squared sums added
    def calculate_EAR_path(self, image_path):
        # if you do want to check it with a certain image
        
        if not cv2.os.path.isfile(image_path):
             raise FileNotFoundError(f"Image file not found at {image_path}.")
        image = cv2.imread(image_path)
        
        return self.calculate_EAR(image)
    
    def calculate_EAR(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray,0)

        if len(rects) == 0:
            print("No face detected")
            return None , None
        
        for rect in rects:
            shape = self.predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:42] # landmarkes in indexes 37 - 42
            right_eye = shape[42:48] # landmarkes in indexes 43 - 48

            left_EAR = self._calculate_EAR(left_eye)
            right_EAR = self._calculate_EAR(right_eye)

            return left_EAR, right_EAR

    def _calculate_EAR(self, eye):
        if len(eye) != 6:
            raise ValueError(f"Expected 6 eye landmarks, got {len(eye)}.")
        
        a = self._distance(eye[1] , eye[5])
        b = self._distance(eye[2], eye[4])
        c = self._distance(eye[0], eye[3])
        
        ear = (a + b)/ (2*c)

        return ear
    
    @staticmethod
    def is_eye_closed(ear, threshold = 0.19):
        #The threshold is good enough
        # Determines if the eye is closed based on EAR.
        return ear <= threshold


        

        