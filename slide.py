import pandas as pd
import numpy as np

class Slides:
    def __init__(self, blink_df_path, gaze_df_path, user_inputs_path, word_count, velocity_threshold):
        self.blink_df = pd.read_csv(blink_df_path)
        self.gaze_df = pd.read_csv(gaze_df_path)
        self.user_inputs = pd.read_csv(user_inputs_path)
        self.word_count = word_count
        self.velocity_threshold = velocity_threshold
        self.slides = self.get_slides()
       

    def get_slides(self):
        slides = []
        for slide in self.number_of_slides(self.user_inputs):
            blink_df = self.blink_df[self.blink_df['slide'] == slide]
            gaze_df = self.gaze_df[self.gaze_df['slide'] == slide]

            slides.append(Slide(slide,blink_df, gaze_df, self.user_inputs, self.word_count[slide], self.velocity_threshold))
        return slides
    
    @staticmethod
    def number_of_slides(df):
        if 'slide' not in df.columns:
            raise ValueError("The DataFrame must have a 'slide' column.")
        return df['slide'].unique()
    
    def mind_wandering_report(self):
        report_data = []
        for slide in self.slides:
            mw_flag, metrics = slide.detect_mind_wandering_overall()
            report_data.append({
                'slide': slide.slide_number,
                'time_period': slide.slide_period,
                'mind_wandering': mw_flag,
                'velocity_flag': metrics['velocity_metrics'],
                'blink_rate_flag': metrics['blink_rate_metrics'],
                'blink_duration_flag': metrics['blink_duration_metrics'],
            })
            #print(metrics)
        
        report_df = pd.DataFrame(report_data)
        return report_df

class Slide:
    def __init__(self, slide_number, blink_df, gaze_df, user_inputs_path, word_count, velocity):
        self.slide_number = slide_number
        self.blink_df = blink_df
        self.gaze_df = gaze_df
        self.slide_period = self.slide_times(user_inputs_path, self.slide_number)
        self.word_count = word_count
        self.estimeated_reading_time = word_count / (self.slide_period[1] - self.slide_period[0])
        self.velocity_threshold = velocity


    @staticmethod
    def slide_times(user_inputs_df, slide_number):
        slide_df = user_inputs_df[user_inputs_df['action'] == 'slide_transition']
        slide_periods = slide_df['time'].values

        if slide_number == 0:
            return 0, slide_periods[0]
        elif slide_number == len(slide_periods):
            return slide_periods[-1], None
        else:
            return slide_periods[slide_number - 1], slide_periods[slide_number]
        
    def get_blink_rate(self):
        number_of_blinks = len(self.blink_df)
        slide_duration = self.slide_period[1] - self.slide_period[0]
        return number_of_blinks/slide_duration if slide_duration != 0 else 0
    
    def get_avg_blink_duration(self):
        print(self.blink_df)
        return self.blink_df['duration'].mean()
        
    def extract_fixation_features(self):
        gaze_df = self.gaze_df.copy()

        # Now compute the average x position
        gaze_df['x_pos'] = (gaze_df['left_eye_x'] + gaze_df['right_eye_x']) / 2.0 #crashes here

        #optional smoothing
        #gaze_df['x_pos_smoothed'] = gaze_df['x_pos'].rolling(window=5, min_periods=1, center=True).mean()

        gaze_df['time'] = gaze_df['start_time']

        gaze_df = gaze_df.sort_values(by = 'time').reset_index(drop=True)

        x_positions = gaze_df['x_pos'].values
        time = gaze_df['time'].values

        velocity = np.zeros(len(x_positions))
        # Central difference for interior points
        for i in range(1, len(x_positions) - 1):
            dt = time[i+1] - time[i-1]
            velocity[i] = (x_positions[i+1] - x_positions[i-1]) / dt if dt != 0 else 0
        
        # Handle the boundaries (e.g., using forward/backward difference)
        if len(velocity) > 1:
            velocity[0] = velocity[1]
            velocity[-1] = velocity[-2]

        gaze_df['velocity'] = velocity
        # Compute fixation
        gaze_df['fixation'] = np.abs(gaze_df['velocity']) < self.velocity_threshold

        fixation_duration = []
        avg_velocity = []
        start_idx = None

        for i, is_fix in enumerate(gaze_df['fixation']):
            if is_fix and start_idx is None:
                start_idx = i
            elif not is_fix and start_idx is not None:
                duration = gaze_df.loc[i - 1, 'time'] - gaze_df.loc[start_idx, 'time']
                fixation_duration.append(duration)
                
                avg_v = gaze_df.loc[start_idx:i, 'velocity'].mean()
                avg_velocity.append(avg_v)
                
                start_idx = None
        if start_idx is not None:
            duration = gaze_df.loc[i, 'time'] - gaze_df.loc[start_idx, 'time']
            fixation_duration.append(duration)
            avg_v = gaze_df.loc[start_idx:len(gaze_df), 'velocity'].mean()
            avg_velocity.append(avg_v)
        
        return np.array(fixation_duration), np.array(avg_velocity)
    
    def detect_mind_wandering_velocity(self, erratic_ratio_threshold=0.3):
        fixation_duration, avg_velocity = self.extract_fixation_features()
        
        erratic_velocity_threshold = self.velocity_threshold * 1.5

        if len(fixation_duration) == 0:
            return False, {}
        erratic_fixations = avg_velocity > erratic_velocity_threshold
        erratic_ratio = np.sum(erratic_fixations) / len(fixation_duration)

        mind_wandering = erratic_ratio > erratic_ratio_threshold

        metrics = {
        'total_fixations': len(avg_velocity),
        'erratic_fixations': int(np.sum(erratic_fixations)),
        'erratic_ratio': erratic_ratio,
        'avg_fixation_velocities': avg_velocity     
        }

        return mind_wandering, metrics
    
    def detect_mind_wandering_blink(self, blink_rate_threshold=0.5, 
                                 avg_blink_duration_threshold_lower_bound=0.1,
                                 avg_blink_duration_threshold_upper_bound=0.4):
        blink_rate = self.get_blink_rate()
        avg_blink_duration = self.get_avg_blink_duration()
        print(blink_rate, avg_blink_duration)

        # Separate flags:
        rate_flag = blink_rate > blink_rate_threshold
        duration_flag = (
            avg_blink_duration < avg_blink_duration_threshold_lower_bound or 
            avg_blink_duration > avg_blink_duration_threshold_upper_bound
        )

        blink_metrics = {
            'blink_rate': blink_rate,
            'avg_blink_duration': avg_blink_duration,
            'rate_flag': rate_flag,
            'duration_flag': duration_flag
        }

        return rate_flag, duration_flag, blink_metrics

    def detect_mind_wandering_overall(self):
        v_flag , v_metrics = self.detect_mind_wandering_velocity()
        br_flag, bd_flag, b_metrics = self.detect_mind_wandering_blink()
        #print (self.slide_number, v_flag, b_flag)

        mind_wandering = v_flag or br_flag or bd_flag
    
        metrics = {
            'velocity_metrics': v_flag,
            'blink_rate_metrics': br_flag,
            'blink_duration_metrics': bd_flag
        }
        print(metrics)
        return mind_wandering, metrics




if __name__ == "__main__":
    blink_df_path = "data/test/processed_data/video_recording_test1_blink_log.csv"
    gaze_df_path = "data/test/processed_data/video_recording_test1_gaze_log.csv"
    user_inputs_path = "data/test/user_inputs_test1.csv" 

    word_count = [100, 100, 100, 100, 100, 100, 100]
    slides = Slides(blink_df_path, gaze_df_path, user_inputs_path, word_count)

    report_df = slides.mind_wandering_report()
    #print(metrics)
    report_df.to_csv("data/test/mind_wandering_report.csv", index=False)