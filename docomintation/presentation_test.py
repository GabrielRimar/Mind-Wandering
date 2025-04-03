import cv2
import time
import pandas as pd
from pynput import keyboard
import threading
import os
import argparse
from face_detection import Face

class KeyboardListener:
    def __init__(
        self,
        file_name,
        number_of_slides,
        key_to_track='p',
        stop_key='q',
        calibration_key='k',
        stop_event=None,
        video_time_source=None  # <--- new parameter
    ):
        self.log_file = pd.DataFrame(columns=['action', 'time', 'slide'])
        self.key_to_track = key_to_track
        self.stop_key = stop_key
        self.calibration_key = calibration_key
        self.stop_event = stop_event
        self.file_name = file_name + '.csv'
        self.slide = 0
        self.number_of_slides = number_of_slides
        
        # Instead of tracking real time, we rely on an external video time source
        self.video_time_source = video_time_source

        # self.start_time = time.time()  # (We no longer need this if we rely on video_time_source)
        self.calibrating = False
        self.listener = None

        print("[DEBUG] KeyboardListener initialized with video time source.")

    def on_press(self, key):
        """
        Record user input times based on the shared video time
        from video_time_source.get_current_video_time().
        """
        # 1. Grab the current video time
        if self.video_time_source is not None:
            current_time = self.video_time_source.get_current_video_time()
        else:
            # fallback - if no source provided, maybe use real clock (not ideal)
            current_time = 0

        try:
            if hasattr(key, 'char') and key.char is not None:
                if key.char == self.key_to_track:
                    self.log_file = pd.concat([
                        self.log_file, 
                        pd.DataFrame({
                            'action': ['mind_wandering'],
                            'time': [current_time],
                            'slide': [self.slide]
                        })
                    ])

                elif key.char == self.calibration_key:
                    action = 'start_calibrating' if not self.calibrating else 'end_calibrating'
                    self.log_file = pd.concat([
                        self.log_file, 
                        pd.DataFrame({
                            'action': [action],
                            'time': [current_time],
                            'slide': [self.slide]
                        })
                    ])
                    self.calibrating = not self.calibrating

                elif key.char == self.stop_key:
                    self.stop()
            
            elif key == keyboard.Key.space:
                self.log_file = pd.concat([
                    self.log_file, 
                    pd.DataFrame({
                        'action': ['slide_transition'],
                        'time': [current_time],
                        'slide': [self.slide]
                    })
                ])
                self.slide += 1
                if self.slide >= self.number_of_slides:
                    self.stop()

        except AttributeError:
            # Handling for special keys
            if key == keyboard.Key.space:
                self.log_file = pd.concat([
                    self.log_file, 
                    pd.DataFrame({
                        'action': ['slide_transition'],
                        'time': [current_time],
                        'slide': [self.slide]
                    })
                ])
                self.slide += 1
                if self.slide >= self.number_of_slides:
                    self.stop()

class WebcamRecorder:
    def __init__(self, output_file, source, stop_event):
        self.output_file = output_file
        self.source = source
        self.stop_event = stop_event
        self.face_detector = Face(shape_predictor_path)
        
        self.recording_started = False
        self.latest_frame = None
        self.current_video_time = 0  # We'll update this every loop
        self.fps = 60  # or whatever you actually get from the camera
        
        print("[DEBUG] WebcamRecorder initialized.")

    def get_current_video_time(self):
        """
        Returns the current video time in seconds, 
        computed as frame_count / fps.
        """
        return self.current_video_time

    def start_recording(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print("[ERROR] Could not open webcam! Check camera permissions.")
            return

        print("[DEBUG] Webcam opened successfully. Warming up camera...")
        
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                print("[DEBUG] Camera warmed up successfully.")
                break
            time.sleep(0.1)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # You can still detect real FPS from your camera if you want:
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            # Fallback if camera doesn't report properly
            self.fps = 30.0  

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_file, fourcc, self.fps, (frame_width, frame_height))

        frame_count = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame! Retrying...")
                time.sleep(0.1)
                continue

            # Update current video time using frame count
            self.current_video_time = frame_count / self.fps
            frame_count += 1

            highlighted_frame = self.show_highlighted_face(frame)
            self.latest_frame = highlighted_frame

            out.write(frame)

        cap.release()
        out.release()
        print("[DEBUG] Webcam recording stopped.")

def ensure_directories_exist():
    os.makedirs(os.path.dirname(video_file), exist_ok=True)
    os.makedirs(os.path.dirname(inputs_file), exist_ok=True)
def run_check():
    global video_file, inputs_file
    number_of_slides = 7
    stop_event = threading.Event()

    ensure_directories_exist()

    webcam_recorder = WebcamRecorder(output_file=video_file, source=0, stop_event=stop_event)
    
    # Pass the webcam_recorder to the listener so it can grab video time
    keyboard_listener = KeyboardListener(
        file_name=inputs_file,
        number_of_slides=number_of_slides,
        key_to_track='p',
        stop_key='q',
        calibration_key='k',
        stop_event=stop_event,
        video_time_source=webcam_recorder
    )

    webcam_thread = threading.Thread(target=webcam_recorder.start_recording)
    keyboard_thread = threading.Thread(target=keyboard_listener.start)

    webcam_thread.start()
    keyboard_thread.start()

    # The main loop ...
    while not stop_event.is_set():
        if webcam_recorder.latest_frame is not None:
            try:
                if threading.current_thread() is threading.main_thread():
                    cv2.imshow("webcam detections", webcam_recorder.latest_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[DEBUG] 'q' key pressed. Stopping recording.")
                        stop_event.set()
                        break
            except Exception as e:
                print(f"[ERROR] Display frame error: {e}")
        time.sleep(0.01)

    stop_event.set()
    webcam_thread.join()
    keyboard_listener.stop()
    keyboard_thread.join()

    cv2.destroyAllWindows()
    print("[DEBUG] Recording and logging completed.")


parser = argparse.ArgumentParser(description="Run the recording and processing pipeline.") 
parser.add_argument("-f", "--folder", type=str, required=True, help="Folder path to save files") 
parser.add_argument("-s", "--subject", type=str, required=True, help="Test subject name") 
parser.add_argument("-shape_predictor", "--shape_predictor", type=str, required=True, help="Shape predictor file path")

args = parser.parse_args()

def get_full_path(folder_path, file_name):
    return os.path.join(folder_path, file_name)

video_file = get_full_path(args.folder, f'video_recording_{args.subject}.mp4')  
inputs_file = get_full_path(args.folder, f'user_inputs_{args.subject}')
shape_predictor_path = args.shape_predictor

run_check()
