import threading
import time
import cv2
import pandas as pd
from pynput import keyboard

class KeyboardListener:
    def __init__(self, ear_calibration_key, reading_calibration_key, start_time, stop_key='q', stop_event=None, number_of_slides=2):
        self.ear_calibration_key = ear_calibration_key  # Fixed typo here
        self.reading_calibration_key = reading_calibration_key
        self.stop_key = stop_key
        self.number_of_slides = number_of_slides
        self.start_time = start_time
        self.stop_event = stop_event if stop_event is not None else threading.Event()

        # Initialize calibration flags
        self.EAR_calibration = False
        self.reading_calibration = False

        # Create the pynput keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)

    def start(self):
        self.listener.start()

    def on_press(self, key):
        current_time = time.time() - self.start_time

        try:
            if hasattr(key, 'char') and key.char is not None:
                if key.char == self.ear_calibration_key:
                    print("Ear calibration key pressed.")
                    self.EAR_calibration = True
                elif key.char == self.reading_calibration_key:
                    print("Reading calibration key pressed.")
                    self.reading_calibration = True
                elif key.char == self.stop_key:
                    print("Stop key pressed.")
                    self.stop_event.set()
            elif key == keyboard.Key.space:
                print("Space key pressed. Slide transition.")
                self.number_of_slides -= 1
                if self.number_of_slides <= 0:
                    print("End of calibration")
                    self.stop_event.set()
        except AttributeError:
            # Fallback in case of an attribute error
            if key == keyboard.Key.space:
                print("Space key pressed. Slide transition.")
                self.number_of_slides -= 1
                if self.number_of_slides <= 0:
                    print("End of calibration")
                    self.stop_event.set()


class Calibration:
    def __init__(self):
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.keyboard_listener = KeyboardListener(
            ear_calibration_key='k',
            reading_calibration_key='l',
            start_time=self.start_time,
            stop_key='q',
            stop_event=self.stop_event
        )

    def start_calibration(self, shape_predictor_path = "shape_predictor_68_face_landmarks.dat"):
        # Import external modules (ensure these are in your PYTHONPATH)
        from face_detection import Face
        from gaze_analysis import GazeAnalysis

        detected_face = Face(shape_predictor_path)
        gaze_log = GazeAnalysis()

        # Start the keyboard listener thread
        self.keyboard_listener.start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Can't open video source")

        ear_values = []
        start_EAR_calibration = False
        start_velocity_calibration = False

        left_eye_vector = None
        right_eye_vector = None
        gaze_start_time = None

        # Main loop
        try:
            while not self.keyboard_listener.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Calibration", frame)
                current_time = time.time()

                # Check for stop event from both cv2 and keyboard listener
                if self.keyboard_listener.stop_event.is_set():
                    print("Stop event triggered. Exiting calibration loop.")
                    break
                face_detected = False
                try:
                    detected_face.refresh(frame)
                    face_detected = True
                except ValueError as e:
                    face_detected = False
                
                def avg_ear_value():
                    ear_value = (detected_face.right_eye.ear + detected_face.left_eye.ear) / 2
                    return ear_value
                    
                # EAR calibration
                if self.keyboard_listener.EAR_calibration and not start_EAR_calibration and face_detected:
                    self.keyboard_listener.EAR_calibration = False
                    start_EAR_calibration = True
                    ear_values.append(avg_ear_value())
                elif start_EAR_calibration and face_detected:
                    ear_values.append(avg_ear_value())

                # (Optional) Reset EAR calibration if needed:
                elif self.keyboard_listener.EAR_calibration and start_EAR_calibration and face_detected:
                    self.keyboard_listener.EAR_calibration = False
                    start_EAR_calibration = False

                # Gaze tracking calibration
                if (self.keyboard_listener.reading_calibration and not start_velocity_calibration) and face_detected:
                    self.keyboard_listener.reading_calibration = False
                    start_velocity_calibration = True
                    new_left_eye_vector, new_right_eye_vector = detected_face.gaze_detection()
                    if (left_eye_vector, right_eye_vector) != (new_left_eye_vector, new_right_eye_vector):
                        left_eye_dim = detected_face.left_eye.frame.shape[:2]
                        right_eye_dim = detected_face.right_eye.frame.shape[:2]
                        # For the first point, using current_time for both start and end
                        gaze_log.add_point(new_left_eye_vector, new_right_eye_vector, current_time, current_time, left_eye_dim, right_eye_dim)
                        left_eye_vector = new_left_eye_vector
                        right_eye_vector = new_right_eye_vector
                        gaze_start_time = current_time
                
                elif start_velocity_calibration and face_detected:
                    detected_face.refresh(frame)
                    new_left_eye_vector, new_right_eye_vector = detected_face.gaze_detection()
                    if (left_eye_vector, right_eye_vector) != (new_left_eye_vector, new_right_eye_vector):
                        left_eye_dim = detected_face.left_eye.frame.shape[:2]
                        right_eye_dim = detected_face.right_eye.frame.shape[:2]
                        gaze_log.add_point(left_eye_vector, right_eye_vector, gaze_start_time, current_time, left_eye_dim, right_eye_dim)
                        left_eye_vector = new_left_eye_vector
                        right_eye_vector = new_right_eye_vector
                        gaze_start_time = current_time
                
                elif self.keyboard_listener.reading_calibration and start_velocity_calibration :
                    self.keyboard_listener.reading_calibration = False
                    start_velocity_calibration = False
                    # Optionally save calibration data here

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.keyboard_listener.stop_event.set()
                    break

        # Cleanup
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            # STOP & JOIN the calibration listener thread
            if self.keyboard_listener.listener is not None:
                self.keyboard_listener.listener.stop()
                self.keyboard_listener.listener.join()

            # Signal just in case
            self.keyboard_listener.stop_event.set()

        # small pause to let everything settle
            time.sleep(0.2)

        self.ear_values = pd.Series(ear_values)
        self.gaze_df = gaze_log.gaze_df

    def data_process(self):
        # Placeholder for data processing implementation
        avg_ear = self.ear_values.mean()

        self.gaze_df['left_eye_x'] = self.gaze_df['left_eye_from_center'].apply(lambda x: x[0])
        self.gaze_df['right_eye_x'] = self.gaze_df['right_eye_from_center'].apply(lambda x: x[0])

        # Ensure a time column exists by copying from 'start_time'
        if 'time' not in self.gaze_df.columns:
            self.gaze_df['time'] = self.gaze_df['start_time']
        def compute_avg_velocity():
            gaze_df = self.gaze_df.copy()
            import numpy as np
        
        # Ensure a time column exists
            if 'start_time' not in gaze_df.columns:
                raise ValueError("Gaze DataFrame missing 'start_time' column")
            
            gaze_df['x_pos'] = (gaze_df['left_eye_x'] + gaze_df['right_eye_x']) / 2.0

            gaze_df = gaze_df.sort_values(by='time').reset_index(drop=True)
            
            x_positions = gaze_df['x_pos'].values
            times = gaze_df['time'].values
            velocity = np.zeros(len(x_positions))
            
            #Compute central differences for interior points.
            for i in range(1, len(x_positions) - 1):
                dt = times[i + 1] - times[i - 1]
                velocity[i] = (x_positions[i + 1] - x_positions[i - 1]) / dt if dt != 0 else 0
            # Handle boundary values
            if len(velocity) > 1:
                velocity[0] = velocity[1]
                velocity[-1] = velocity[-2]
            
            gaze_df['velocity'] = velocity        
            # Return the average absolute velocity (or customize as needed)
            avg_velocity = np.mean(np.abs(velocity))

            return avg_velocity
        
        avg_velocity = compute_avg_velocity()
        
        
        #Return of the data processing
        return avg_ear, avg_velocity
    
import argparse    
from presentation_handler import PresentationHandler
import os

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Run face/EAR calibration and save to CSV")
    p.add_argument("-f", "--folder",required=True,  help="Folder to write calibration_values_{subject}.csv")
    p.add_argument("-s", "--subject",required=True,  help="Subject ID, used in file name")
    p.add_argument("-sp","--shape_predictor",required=True,  help="Path to shape_predictor_68_face_landmarks.dat")
    args = p.parse_args()


    calib = Calibration()
    presentation = PresentationHandler("presentation/calibration.pptx")
    presentation.open_presentation()
    calib.start_calibration()
    # After calibration, you can call calib.data_process() or save results as needed.
    
    avf_ear, avg_velocity = calib.data_process()
    # Save the results to a CSV file
    output_dir  = args.folder
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir,
                      f"calibration_values_{args.subject}.csv")

    data = {
        'avg_ear': [avf_ear],
        'avg_velocity': [avg_velocity]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Calibration values saved to {output_file}")
    
    
