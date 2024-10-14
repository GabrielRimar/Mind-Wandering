# import the necessary packages
import os
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

from EAR_calculation import EARCalculator
from blink_analysis import BlinkAnalysis

predictor_path = "shape_predictor_68_face_landmarks.dat"

ear_calculator = EARCalculator(predictor_path)

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()


# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Path to the shape predictor model

# Initialize the shape predictor
predictor = dlib.shape_predictor(predictor_path)

blink_start_time = None
blink_log = BlinkAnalysis()
last_check = 0
stopped_blibking = 0
# Loop over frames from the video stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    

    # If frame is not captured properly, break the loop
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=600)
    
    image = frame
    
    left_ear, right_ear = ear_calculator.calculate_EAR(image)
    current_time = blink_log.get_time_from_start()
    if left_ear is not None and right_ear is not None:
        #print(f"Left EAR: {left_ear:.3f} | Right EAR: {right_ear:.3f}")

        left_closed = ear_calculator.is_eye_closed(left_ear)
        right_closed = ear_calculator.is_eye_closed(right_ear)


        if (left_closed or right_closed) and blink_start_time is None:
            blink_start_time = current_time
        
        elif blink_start_time is not None and not left_closed and not right_closed:
            if(stopped_blibking != 0):
                if(current_time - stopped_blibking > 0.001):# time for the program to notice another blink
                    blink_end_time = stopped_blibking

                    blink_log.add_blink(blink_start_time , blink_end_time)

                    #print(blink_log[len(blink_log) -1])
                    blink_start_time = None
                    stopped_blibking = 0
                    #print("Blink detected")
                    #print(f"Blink duration {blink_duration}")
            else:
                stopped_blibking = current_time
        else:
            stopped_blibking = 0
            
    
    if(current_time - last_check >= 5):
        print(f"blink rate per minute: {blink_log.get_blinks_last_minute()}")
        print(f"blink avg duration in the last minute: {blink_log.get_avg_blink_duration_last_minute()}")
        last_check = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Convert dlib's rectangle to an OpenCV-style bounding box (x, y, w, h)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Loop over the (x, y)-coordinates for the facial landmarks and draw them
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
    # Display the resulting frame
    cv2.imshow("Facial Landmarks", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        blink_log.save_data()
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
