import pandas as pd
import os

class GazeAnalysis:
    def __init__(self):
        self.gaze_df = pd.DataFrame(columns=['left_eye_from_center', 'right_eye_from_center', 'start_time', 'end_time', 'left_eye_dim', 'right_eye_dim'])

    def add_point(self, left_eye, right_eye, time_start, time_end, left_eye_dim, right_eye_dim):
        if left_eye is None or right_eye is None:
            return
        new_point = pd.Series({
            'left_eye_from_center' : left_eye,
            'right_eye_from_center' : right_eye,
            'start_time' : time_start,
            'end_time' : time_end,
            'left_eye_dim' : left_eye_dim,
            'right_eye_dim' : right_eye_dim
        })

        self.gaze_df = pd.concat([self.gaze_df, new_point.to_frame().T], ignore_index=True)
    
    def save_data(self, file_name):
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        file_path = os.path.join(file_name)
        self.gaze_df.to_csv(file_path, index=False)