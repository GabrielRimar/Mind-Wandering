import threading

from webcam_recorder import WebcamRecorder
from keypress_listener import KeyboardListener


def run_check():
    stop_event = threading.Event()
    
    # Initialize WebcamRecorder with the output file path and source (0 for default webcam)
    webcam_recorder = WebcamRecorder(output_file='output_video.avi', source=0, stop_event= stop_event)
    
    # Initialize KeyboardListener to track key 'p' for mind-wandering and stop on 'q'
    keyboard_listener = KeyboardListener('test',stop_event= stop_event)
    
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

if __name__ == "__main__":
    run_check()
