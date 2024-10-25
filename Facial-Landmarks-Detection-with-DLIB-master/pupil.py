import numpy as np
import cv2


class Pupil:
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame):
        self.iris_frame = None
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame):
        """Performs operations on the eye frame to isolate the iris"""
        # Convert to grayscale
        eye_frame_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average intensity
        non_black_pixels = eye_frame_gray[eye_frame_gray > 0]
        avg_intensity = np.mean(non_black_pixels)

        # Set all 100% black pixels to white
        eye_frame_gray[eye_frame_gray == 0] = 255

        
        # Apply thresholding based on the average intensity
        _, thresholded_frame1 = cv2.threshold(eye_frame_gray, avg_intensity, 255, cv2.THRESH_BINARY)
        _, thresholded_frame2 = cv2.threshold(eye_frame_gray, avg_intensity + 10, 255, cv2.THRESH_BINARY)

        combined_frame = cv2.bitwise_and(thresholded_frame1, thresholded_frame2)

        combined_frame = cv2.bitwise_not(combined_frame)

        return combined_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by calculating the centroid."""
        self.iris_frame = self.image_processing(eye_frame)

        # Find contours in the processed frame
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # Sort contours by area and choose the largest one, assuming it's the iris
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        try:
            moments = cv2.moments(contours[0])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass