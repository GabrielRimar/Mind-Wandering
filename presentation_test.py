import cv2
import time
import pandas as pd
from pynput import keyboard

class KeyboardListener:
    def __init__(self,file_name, number_of_slides ,key_to_track='p', stop_key='q', calibration_key = 'k',stop_event = None):
        self.log_file = pd.DataFrame(columns=['action', 'timestamp', 'slide'])
        self.key_to_track = key_to_track
        self.stop_key = stop_key
        self.stop_event = stop_event
        self.file_name = file_name + '.csv'
        self.slide = 0 # the slide i am on
        self.number_of_slides = number_of_slides
        self.calibration_key = calibration_key
        self.start_time = time.time()

        #self.calibration_times = []
        self.calibrating = False
        
        
    def on_press(self, key):
        current_time = time.time() - self.start_time
        try:
            if key == self.key_to_track:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['mind_wandering'], 'time': [current_time], 'slide' : self.slide})])
            
            elif key.char == self.stop_key and self.stop_event:
                self.stop()
            
            elif key.char == self.calibration_key and not self.calibrating:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['start_calibrating'], 'time': [current_time], 'slide' : self.slide})])
                #self.calibration_times.append((current_time, 'start'))
                self.calibrating = True
            
            elif key.char == self.calibration_key and self.calibrating:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['end_calibrating'], 'time': [current_time], 'slide' : self.slide})])
                #self.calibration_times.append((current_time, 'end'))
                self.calibrating = False

        except AttributeError:
            if key == keyboard.Key.space:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['slide_transition'], 'time': [current_time], 'slide' : self.slide})])
                self.slide += 1
                if self.slide >= self.number_of_slides:
                    self.stop()
    
    def start(self):
        listener = keyboard.Listener(on_press= self.on_press)
        listener.start()

    def stop(self):
        self.stop_event.set()
        self.log_file.to_csv(self.file_name, index=False)
        print({self.file_name})


from face_detection import Face

class WebcamRecorder:
    def __init__(self, output_file, source, stop_event):
        global shape_predictor_path
        self.output_file = output_file
        self.source = source
        self.stop_event = stop_event
        self.face_detector = Face(shape_predictor_path)
        self.recording_started = False

    


    def start_recording(self, show_frame = True):
        cap = cv2.VideoCapture(self.source)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_file, fourcc, 47, (640, 480))

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                if not self.start_recording:
                    self._find_face(frame)
                else: 
                    if show_frame:
                        self.show_highlighted_face(frame)
                    out.write(frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def show_highlighted_face(self, frame):
        if frame is None or frame.size == 0:
            return
        
        try:
            self.face_detector.refresh(frame)
            cv2.imshow("webcam detections", self.face_detector.highlight_landmarks())
        except ValueError as e:
            print(e)

    def _find_face(self, frame):
        try:
            self.face_detector.refresh(frame)
            self.start_recording = True
        except ValueError as e:
            print(e)
        

    def stop_recording(self):
        self.stop_recording = True
    
    


import threading
import os
import argparse

def run_check():
    global video_file
    global inputs_file
    number_of_slides = 2

    stop_event = threading.Event()
    
    
    # Initialize WebcamRecorder with the output file path and source (0 for default webcam)
    webcam_recorder = WebcamRecorder(output_file= video_file, source=0, stop_event= stop_event)
    
    # Initialize KeyboardListener to track key 'p' for mind-wandering and stop on 'q'
    keyboard_listener = KeyboardListener(inputs_file, number_of_slides ,stop_event= stop_event)
    
    # Start webcam recording in a separate thread
    webcam_thread = threading.Thread(target=webcam_recorder.start_recording)
    
    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener.start)
    
    # Start both threads
    webcam_thread.start()
    keyboard_thread.start()

    # Join the threads to make sure they both run simultaneously
    webcam_thread.join()
    keyboard_thread.join()

    #if key 'q' pressed make both stop

parser = argparse.ArgumentParser(description="Run the recording and processing pipeline.") 
parser.add_argument("-f", "--folder", type=str, required=True, help="Folder path to save files") 
parser.add_argument("-s", "--subject", type=str, required=True, help="Test subject name") 
parser.add_argument("-shape_predictor", "--shape_predictor", type=str, required=True, help="Shape predictor file path")

args = parser.parse_args()

def get_full_path(folder_path, file_name):
    return os.path.join(folder_path, file_name)

video_file = get_full_path(args.folder, f'video_recording_{args.subject}.avi')
inputs_file = get_full_path(args.folder, f'user_inputs_{args.subject}')
shape_predictor_path = args.shape_predictor
blink_log = get_full_path(args.folder, f'blink_log_{args.subject}')
gaze_log = get_full_path(args.folder, f'gaze_log_{args.subject}')

run_check()


