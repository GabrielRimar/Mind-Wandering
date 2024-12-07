import cv2

from face_detection import Face

class WebcamRecorder:
    def __init__(self, output_file, source, stop_event, shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'):
        self.output_file = output_file
        self.source = source
        self.stop_event = stop_event
        self.face_detector = Face(shape_predictor_path)
        self.recording_started = False

    def start_recording(self):
        cap = cv2.VideoCapture(self.source)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_file, fourcc, 25.0, (640, 480))

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                if not self.start_recording:
                    self._find_face(frame)
                else: 
                    out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def _find_face(self, frame):
        try:
            self.face_detector.refresh(frame)
            self.start_recording = True
        except ValueError as e:
            print(e)
        

    def stop_recording(self):
        self.stop_recording = True