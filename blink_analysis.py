import pandas as pd
import os

class BlinkAnalysis:
    def __init__(self):
        self.blink_df = pd.DataFrame(columns=['start_time', 'end_time', 'duration'])
    
    def add_blink(self, start_time, end_time):
        duration = self._blink_duration(start_time, end_time)

        new_blink = pd.Series({
            'start_time' : start_time,
            'end_time' : end_time,
            'duration' : duration
        })
        self.blink_df = pd.concat([self.blink_df, new_blink.to_frame().T], ignore_index=True)
    
    @staticmethod
    def _blink_duration(start_time, end_time):
        return end_time - start_time
    
    
    def get_blinks_last_minute(self):
        #method will return the blinking rate in the last minute or less if the info isn't aveliable
        last_min = self._get_df_last_minute()
        blink_count = last_min.shape[0]
        if(blink_count == 0):
            return 0
        
        time_passed = self.get_time_from_start() - last_min['start_time'].min()
        
        time_passed = time_passed / 60.0

        return blink_count / time_passed
    
    def get_avg_blink_duration_last_minute(self):
        last_min = self._get_df_last_minute()
        
        if not last_min.empty:
            return last_min['duration'].mean()
        else: 
            return 0
      
    def _get_df_last_minute(self):
        current_time = self.get_time_from_start()
        last_min = self.blink_df[self.blink_df['start_time'] > current_time - 60]
        return last_min
    
    def save_data(self, file_name):
        file_path = os.path.join(file_name  + '.csv')
        self.blink_df.to_csv(file_path, index=False)