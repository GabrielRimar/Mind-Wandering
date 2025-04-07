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
        video_time_source=None
    ):
        """
        KeyboardListener now uses a 'video_time_source' (typically a WebcamRecorder)
        to retrieve the current video time instead of real clock time.
        """
        self.log_file = pd.DataFrame(columns=['action', 'time', 'slide'])
        self.key_to_track = key_to_track
        self.stop_key = stop_key
        self.calibration_key = calibration_key
        self.stop_event = stop_event
        self.file_name = file_name + '.csv'
        self.slide = 0
        self.number_of_slides = number_of_slides

        # Provide the same time reference used by the WebcamRecorder
        self.video_time_source = video_time_source

        # We no longer need start_time, because we'll rely on the video time
        self.calibrating = False
        self.listener = None

        print("[DEBUG] KeyboardListener initialized with video time source.")

    def on_press(self, key):
        """
        Called whenever a key is pressed. Logs events using video time from 'video_time_source'.
        """
        if self.video_time_source is not None:
            current_time = self.video_time_source.get_current_video_time()
        else:
            # Fallback if no source is provided (not ideal, but avoids crashes)
            current_time = 0

        try:
            if hasattr(key, 'char') and key.char is not None:
                if key.char == self.key_to_track:
                    # Mind-wandering key
                    self.log_file = pd.concat([
                        self.log_file,
                        pd.DataFrame({
                            'action': ['mind_wandering'],
                            'time': [current_time],
                            'slide': [self.slide]
                        })
                    ])

                elif key.char == self.calibration_key:
                    # Toggle calibration start/end
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
                    # Stop the entire process
                    self.stop()
            
            elif key == keyboard.Key.space:
                # Slide transition
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
            # Handling special keys if needed
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

    def start(self):
        """Starts the keyboard listener in a separate thread."""
        print("[DEBUG] Starting KeyboardListener...")
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def stop(self):
        """Stops the listener, saves the log to disk, and signals the main thread if needed."""
        if self.stop_event:
            self.stop_event.set()
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        self.log_file.to_csv(self.file_name, index=False)
        print(f"[DEBUG] Keyboard log saved to {self.file_name}")
        if self.listener:
            self.listener.stop()

class WebcamRecorder:
    def __init__(self, output_file, source, stop_event):
        self.output_file = output_file
        self.source = source
        self.stop_event = stop_event
        self.face_detector = Face(shape_predictor_path)
        
        self.recording_started = False
        self.latest_frame = None
        self.current_video_time = 0  # We'll update this each frame
        self.fps = 60  # default, or overwritten once we read from camera

        print("[DEBUG] WebcamRecorder initialized.")

    def get_current_video_time(self):
        """
        Returns the current video time in seconds,
        computed as frame_count / fps in 'start_recording()'.
        """
        return self.current_video_time

    def start_recording(self):
        """Continuously reads from the webcam, updates current video time, and writes frames to disk."""
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

        # Read the actual FPS from the capture device (if it reports valid data)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps
        else:
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

            # Update current video time: total frames so far / fps
            self.current_video_time = frame_count / self.fps
            frame_count += 1

            highlighted_frame = self.show_highlighted_face(frame)
            self.latest_frame = highlighted_frame
            out.write(frame)

        cap.release()
        out.release()
        print("[DEBUG] Webcam recording stopped.")

    def show_highlighted_face(self, frame):
        """
        Runs face detection on the current frame, draws landmarks if found.
        """
        if frame is None or frame.size == 0:
            return frame
        try:
            self.face_detector.refresh(frame)
            return self.face_detector.highlight_landmarks()
        except ValueError as e:
            print(f"[ERROR] Face detection failed: {e}")
            return frame

def ensure_directories_exist():
    os.makedirs(os.path.dirname(video_file), exist_ok=True)
    os.makedirs(os.path.dirname(inputs_file), exist_ok=True)

from calibration import Calibration
def run_check():
    global video_file, inputs_file
    number_of_slides = 9
    stop_event = threading.Event()

    ensure_directories_exist()
    # Initialize the calibration process
    calibration = Calibration()
    calibration.start_calibration()
    # Wait for calibration to finish
    
    # Create the WebcamRecorder (this will produce the 'video time')
    webcam_recorder = WebcamRecorder(output_file=video_file, source=0, stop_event=stop_event)
    
    # Pass webcam_recorder to KeyboardListener so it can read the same 'video time'
    keyboard_listener = KeyboardListener(
        file_name=inputs_file,
        number_of_slides=number_of_slides,
        stop_event=stop_event,
        video_time_source=webcam_recorder
    )

    # Start threads
    webcam_thread = threading.Thread(target=webcam_recorder.start_recording)
    keyboard_thread = threading.Thread(target=keyboard_listener.start)

    webcam_thread.start()
    keyboard_thread.start()

    # Main loop: display frames until 'q' is pressed or we run out
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

    # Clean up
    stop_event.set()
    webcam_thread.join()
    keyboard_listener.stop()
    keyboard_thread.join()
    
    cv2.destroyAllWindows()
    ear, avg_velocity = calibration.data_process()
   
    file_path = os.path.join(args.folder, f'calibration_values_{args.subject}.csv')
    pd.DataFrame({'ear': [ear], 'avg_velocity': [avg_velocity]}).to_csv(file_path, index=False)
    
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

#qpython3 presentation_test.py -f data/gabriel -s gabriel1 -shape_predictor shape_predictor_68_face_landmarks.dat