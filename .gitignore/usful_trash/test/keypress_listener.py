import time
import threading
import pandas as pd
from pynput import keyboard

class KeyboardListener:
    def __init__(self,file_name, key_to_track='p', stop_key='q', stop_event = None):
        self.log_file = pd.DataFrame(columns=['action', 'timestamp'])
        self.key_to_track = key_to_track
        self.stop_key = stop_key
        self.stop_event = stop_event
        self.file_name = file_name
        
    def on_press(self, key):
        current_time = time.time()
        try:
            if key.char == self.key_to_track:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['mind_wandering'], 'time': [current_time]})])
            elif key.char == self.stop_key and self.stop_event:
                self.stop()
        except AttributeError:
            if key == keyboard.Key.space:
                self.log_file = pd.concat([self.log_file, pd.DataFrame({'action': ['slide_transition'], 'time': [current_time]})])
    
    def start(self):
        listener = keyboard.Listener(on_press= self.on_press)
        listener.start()

    def stop(self):
        self.stop_event.set()
        self.log_file.to_csv(f'{self.file_name}.csv', index=False)