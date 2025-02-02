import pandas as pd
import numpy as np

def load_data(blink_log_path, gaze_log_path, user_inputs_path):
    blink_df = pd.read_csv(blink_log_path)
    gaze_df = pd.read_csv(gaze_log_path)
    user_inputs_df = pd.read_csv(user_inputs_path)
    return blink_df, gaze_df, user_inputs_df

def split_by_slides(blink_df, gaze_df, user_inputs_df):
    slides = user_inputs_df['slide'].unique()
    slide_data = {}
    
    for slide in slides:
        if slide not in user_inputs_df['slide'].values:
            continue
        slide_info = user_inputs_df[user_inputs_df['slide'] == slide]
        if slide_info.empty:
            continue
        slide_start_time = slide_info['time'].iloc[0]
        try:
            slide_end_time = user_inputs_df[user_inputs_df['slide'] == slide]['time'].iloc[1]
        except IndexError:  # No end time available
            slide_end_time = user_inputs_df['time'].max()  # Assume end at last recorded time
        
        slide_blink_df = blink_df[(blink_df['start_time'] >= slide_start_time) & (blink_df['end_time'] <= slide_end_time)]
        slide_gaze_df = gaze_df[(gaze_df['time_start'] >= slide_start_time) & (gaze_df['time_end'] <= slide_end_time)]
        slide_data[slide] = {'blink': slide_blink_df, 'gaze': slide_gaze_df}
    
    return slide_data

def calculate_reading_speed(gaze_df, pixel_threshold=50):
    gaze_df['reading_speed'] = 0
    for i in range(1, len(gaze_df)):
        prev_pos = np.array(eval(gaze_df.iloc[i-1]['right_eye_from_center']))
        curr_pos = np.array(eval(gaze_df.iloc[i]['right_eye_from_center']))
        distance = np.linalg.norm(curr_pos - prev_pos)
        gaze_df.at[i, 'reading_speed'] = 1 if distance > pixel_threshold else 0
    return gaze_df

def process_data(blink_log_path, gaze_log_path, user_inputs_path, output_path):
    blink_df, gaze_df, user_inputs_df = load_data(blink_log_path, gaze_log_path, user_inputs_path)
    slide_data = split_by_slides(blink_df, gaze_df, user_inputs_df)
    
    for slide, data in slide_data.items():
        data['gaze'] = calculate_reading_speed(data['gaze'])
        data['gaze'].to_csv(f"{output_path}/gaze_slide_{slide}.csv", index=False)
        data['blink'].to_csv(f"{output_path}/blink_slide_{slide}.csv", index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process blink and gaze logs to split by slides and calculate reading speed.")
    parser.add_argument("-b", "--blink_log", type=str, required=True, help="Path to the blink log file (CSV)")
    parser.add_argument("-g", "--gaze_log", type=str, required=True, help="Path to the gaze log file (CSV)")
    parser.add_argument("-u", "--user_inputs", type=str, required=True, help="Path to the user inputs file (CSV)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the processed data")

    args = parser.parse_args()

    process_data(args.blink_log, args.gaze_log, args.user_inputs, args.output)
