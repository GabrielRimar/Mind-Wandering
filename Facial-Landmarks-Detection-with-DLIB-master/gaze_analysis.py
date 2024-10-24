import pandas as pd
import os

class GazeAnalysis:
    def __init__(self):
        self.gaze_df = pd.DataFrame(columns=['left_eye_from_center', 'right_eye_from_center', 'time_start', 'time_end'])

    def add_point(self, left_eye, right_eye, time_start, time_end):
        if left_eye is None or right_eye is None:
            return
        new_point = pd.Series({
            'left_eye_from_center' : left_eye,
            'right_eye_from_center' : right_eye,
            'time_start' : time_start,
            'time_end' : time_end
        })

        self.gaze_df = pd.concat([self.gaze_df, new_point.to_frame().T], ignore_index=True)
    
    def save_data(self, file_name):
        folder_path = 'data'
        file_path = os.path.join(folder_path, file_name  + '.csv')

        self.gaze_df.to_csv(file_path, index=False)