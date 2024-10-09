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

from facial_landmarks import detect_facial_landmarks

facial_landmarks_cordinates = detect_facial_landmarks("shape_predictor_68_face_landmarks.dat", "image1.jpg")
facial_landmarks_cordinates = facial_landmarks_cordinates[0].tolist()
#print(facial_landmarks_cordinates)

left_eye = facial_landmarks_cordinates[36:42] # landmarkes in indexes 37 - 42
right_eye = facial_landmarks_cordinates[42:48] # landmarkes in indexes 43 - 48

def distance_between_points(p1, p2):
    if not p1 or not p2:
        raise ValueError("Point 1 or 2 is emty")
    return math.sqrt(math.pow(p1[0] - p2[0],2) +math.pow(p1[1] - p2[1],2))

def EAR(eye):
    if (len(eye) == 6):
        return (distance_between_points(eye[1] , eye[5]) + distance_between_points(eye[2], eye[4]))/(2*distance_between_points(eye[0], eye[3]))
    else:
        raise ValueError("The eye array is not correct in the correct size.\n"
                         "size required 6\n"
                         f"your size {len(eye)}")


#print (f"dddd{left_eye}")

EAR_left = EAR(left_eye)
EAR_Right = EAR(right_eye)

print(f"EAR on left eye: {EAR_left}")

print(f"EAR on right eye: {EAR_Right}")



