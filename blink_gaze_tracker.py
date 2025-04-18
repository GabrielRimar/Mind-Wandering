import cv2
import pandas as pd

from face_detection import Face
from blink_analysis import BlinkAnalysis
from gaze_analysis import GazeAnalysis

class BlinkGazeTracker:
    def __init__(self, shape_predictor_path, blink_log_file_name, gaze_log_file_name, EAR_threshold):
        self.detected_face = Face(shape_predictor_path)

        self.blink_log = BlinkAnalysis()
        self.gaze_log = GazeAnalysis()

        self.blink_log_name = blink_log_file_name
        self.gaze_log_name = gaze_log_file_name
        self.detected_face.set_ear_threshold(EAR_threshold)        
        '''
        try:
            self.analyze_video(video_source)
        except ValueError as e:
            print(e)
        '''

    
    def analyze_video(self, source):

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Can't open video source{source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_time = 1/fps

        blink_start_time = 0
        blink_end = None
        left_eye_vector, right_eye_vector = None, None
        face_detected = False
        
        gaze_start_time = 0
        #blink_start_time = 0
        frame_num = 0
        
        while frame_num < total_frames:
            #current_time = time.time() - self.start_time
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_num * frame_time
            try:
                self.detected_face.refresh(frame)
                face_detected = True
            
            except ValueError as e:
                pass

            # gaze detection part
            if face_detected:
                new_left_eye_vector, new_right_eye_vector = self.detected_face.gaze_detection()
                #print(new_left_eye_vector, new_right_eye_vector)

                if(left_eye_vector, right_eye_vector != new_left_eye_vector, new_right_eye_vector):
                    left_eye_dim = self.detected_face.left_eye.frame.shape[:2]
                    right_eye_dim = self.detected_face.right_eye.frame.shape[:2]
                    self.gaze_log.add_point(left_eye_vector, right_eye_vector, gaze_start_time,current_time, left_eye_dim, right_eye_dim)
                #print(gaze_start_time > current_time)
                    left_eye_vector = new_left_eye_vector
                    right_eye_vector = new_right_eye_vector
                    gaze_start_time = current_time
                
                # Blink detection
                
                #print(self.detected_face.right_eye.ear, self.detected_face.left_eye.ear, self.detected_face.right_eye.threshold)
                left_eye_closed, right_eye_closed = self.detected_face.closed_eyes()
                if blink_start_time == 0 and (left_eye_closed or right_eye_closed):
                    blink_start_time = current_time
                
                elif blink_start_time != 0 and not (left_eye_closed and right_eye_closed):
                    if(blink_end is None):
                        blink_end = current_time
                    if(current_time - blink_end >= 0.05):
                        self.blink_log.add_blink(blink_start_time, current_time)
                        blink_start_time = 0
                        blink_end = None
            frame_num += 1
            
            #cv2.imshow("webcam detections", self.detected_face.highlight_landmarks())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        self.blink_log.save_data(self.blink_log_name)
        self.gaze_log.save_data(self.gaze_log_name)

if __name__ == "__main__":
    shape_predictor = "shape_predictor_68_face_landmarks.dat"
    face_analysis = BlinkGazeTracker(shape_predictor,'test_blink','test_gaze')

    face_analysis.analyze_video('data/video_recording_test.avi', 'data/user_inputs_test.csv')

