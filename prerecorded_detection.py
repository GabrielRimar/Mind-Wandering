import cv2

from face_detection import Face
from blink_analysis import BlinkAnalysis
from gaze_analysis import GazeAnalysis

class BlinkGazeTracker:
    def __init__(self, shape_predictor_path, video_source):
        self.detected_face = Face(shape_predictor_path)

        self.blink_log = BlinkAnalysis()
        self.gaze_log = GazeAnalysis()
        
        try:
            self.analyze_video(video_source)
        except ValueError as e:
            print(e)

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

        #gaze_start_time = 0
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
                print(e)


            # gaze detection part
            if face_detected:
                new_left_eye_vector, new_right_eye_vector = self.detected_face.gaze_detection()

                if(left_eye_vector, right_eye_vector != new_left_eye_vector, new_right_eye_vector):
                    left_eye_dim = self.detected_face.left_eye.frame.shape[:2]
                    right_eye_dim = self.detected_face.right_eye.frame.shape[:2]
                    self.gaze_log.add_point(left_eye_vector, right_eye_vector, gaze_start_time,current_time, left_eye_dim, right_eye_dim)
                    
                    left_eye_vector = new_left_eye_vector
                    right_eye_vector = new_right_eye_vector
                    gaze_start_time = current_time
                
                # Blink detection
                
                left_eye_closed, right_eye_closed = self.detected_face.closed_eyes()
                if blink_start_time == 0 and (left_eye_closed or right_eye_closed):
                    blink_start_time = current_time
                
                elif blink_start_time != 0 and not (left_eye_closed and right_eye_closed):
                    if(blink_end is None):
                        blink_end = current_time
                    if(current_time - blink_end >= 0.01):
                        self.blink_log.add_blink(blink_start_time, current_time)
                        blink_start_time = 0
                        blink_end = None
            
            #cv2.imshow("webcam detections", self.detected_face.highlight_landmarks())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        self.blink_log.save_data('blink_log')
        self.gaze_log.save_data('gaze_log')
