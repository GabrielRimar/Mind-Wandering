import cv2
import imutils
import dlib
from imutils import face_utils

from EAR_calculation import EARCalculator
from blink_analysis import BlinkAnalysis
from video_handler import VideoHandler

class BlinkDetector:
    def __init__(self, predictor_path, video_source, vizualize):
        self.video_handler = VideoHandler(video_source)
        self.ear_calculator = EARCalculator(predictor_path)
        self.blink_log = BlinkAnalysis()
        self.vizualize = vizualize
    
    def detect_blibking(self, resolution = 600):
        blink_start_time = None
        stopped_blinking = 0

        # main loop
        while True:
            ret, frame = self.video_handler.read_frame()
            frame = imutils.resize(frame, width=resolution)
            try:
                left_ear, right_ear = self.ear_calculator.calculate_EAR(frame)
            except ValueError as e:
                print(f"error: {e}")
                #left_ear, right_ear = None

            current_time = self.blink_log.get_time_from_start()

            if left_ear is not None and right_ear is not None:

                left_closed = self.ear_calculator.is_eye_closed(left_ear)
                right_closed = self.ear_calculator.is_eye_closed(right_ear)
                
                if (left_closed and right_closed) and blink_start_time is None:
                    blink_start_time = current_time
                elif blink_start_time is not None and not left_closed and not right_closed:
                    if(stopped_blibking != 0):
                        if(current_time - stopped_blibking > 0.001):# time for the program to notice another blink
                            blink_end_time = stopped_blibking
                            self.blink_log.add_blink(blink_start_time , blink_end_time)
                            
                            blink_start_time = None
                            stopped_blibking = 0
                            
                    else:
                        stopped_blibking = current_time
                else:
                    stopped_blibking = 0
            
            if self.vizualize:
                self.visualisation(frame)

            # Break the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.blink_log.save_data()
                break
        
        self.video_handler.release()
        cv2.destroyAllWindows()

        
    def visualisation(self, frame):
        detector = self.ear_calculator.detector
        predictor = self.ear_calculator.predictor

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the grayscale frame
        rects = detector(gray, 0)
        
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
            



            


if __name__ == "__main__":
    shape_predictor = "shape_predictor_68_face_landmarks.dat"
    source = 0
    detector = BlinkDetector(shape_predictor, source,True)

    detector.detect_blibking()

