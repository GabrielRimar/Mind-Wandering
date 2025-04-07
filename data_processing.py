import os
import pandas as pd
import numpy as np

def run_blink_gaze_tracker(video_file, user_inputs_file, output_folder,file_name, EAR,
                           shape_predictor="shape_predictor_68_face_landmarks.dat"):
    """
    Uses BlinkGazeTracker to process the video and generate blink and gaze logs.
    If the log files already exist, this step is skipped.
    """
    blink_log_path = os.path.join(output_folder, f"{file_name}_blink_log.csv")
    gaze_log_path = os.path.join(output_folder, f"{file_name}_gaze_log.csv")
    
    # Run the tracker only if one or both log files don't exist.
    if not os.path.exists(blink_log_path) or not os.path.exists(gaze_log_path):
        print("Running BlinkGazeTracker to generate logs...")
        from blink_gaze_tracker import BlinkGazeTracker  # Import your tracker class
        tracker = BlinkGazeTracker(shape_predictor, blink_log_path, gaze_log_path , EAR)
        tracker.analyze_video(video_file, user_inputs_file)
    else:
        print("Blink and gaze logs already exist, skipping video processing.")
    
    return blink_log_path, gaze_log_path

def load_data(blink_log_path, gaze_log_path, user_inputs_path):
    """
    Loads blink and gaze logs along with the user inputs CSV.
    """
    
    blink_df = pd.read_csv(blink_log_path) if os.path.exists(blink_log_path) else pd.DataFrame()
    #print("blink_df loaded from:", blink_log_path)
    gaze_df = pd.read_csv(gaze_log_path) if os.path.exists(gaze_log_path) else pd.DataFrame()
    user_inputs_df = pd.read_csv(user_inputs_path)

    import ast

    for col in ['left_eye_from_center', 'right_eye_from_center']:
        if col in gaze_df.columns:
            gaze_df[col] = gaze_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    gaze_df['left_eye_x'] = gaze_df['left_eye_from_center'].apply(lambda p: p[0])
    gaze_df['right_eye_x'] = gaze_df['right_eye_from_center'].apply(lambda p: p[0])

    
    
    #print(gaze_df.head())
    new_blink_df, new_gaze_df = split_by_slides(blink_df, gaze_df, user_inputs_df)

    new_blink_df.to_csv(blink_log_path, index=False)
    new_gaze_df.to_csv(gaze_log_path, index=False)

def split_events_by_slide(events_df, slide_times, session_end=None):
    """
    Splits events (with start_time and end_time) into separate rows if they span slide boundaries.
    Events that do not cross a slide boundary are preserved as-is.
    
    Parameters:
        events_df (pd.DataFrame): DataFrame containing events with 'start_time' and 'end_time'.
        slide_times (list or array): Sorted slide transition times.
        session_end (float, optional): The overall end time of the session. For the last slide, if not provided, it uses infinity.
    
    Returns:
        pd.DataFrame: A new DataFrame with events split by slide intervals.
    """
    # Ensure slide_times are sorted
    slide_times = sorted(slide_times)
    
    def get_slide(time):
        if time < slide_times[0]:
            return 0
        for i in range(len(slide_times) - 1):
            if time >= slide_times[i] and time < slide_times[i + 1]:
                return i + 1
        return len(slide_times)
    
    new_rows = []
    for idx, row in events_df.iterrows():
        event_start = row['start_time']
        event_end = row['end_time']
        start_slide = get_slide(event_start)
        end_slide = get_slide(event_end)
        
        # If the event is entirely within one slide, assign slide number.
        if start_slide == end_slide:
            row_copy = row.copy()
            row_copy['duration'] = event_end - event_start
            row_copy['slide'] = start_slide  # added slide number for single-slide events
            new_rows.append(row_copy)
        else:
            # The event spans one or more slide boundaries.
            boundaries = [t for t in slide_times if event_start < t < event_end]
            segments = [event_start] + boundaries + [event_end]
            for j in range(len(segments) - 1):
                seg_start = segments[j]
                seg_end = segments[j + 1]
                mid = (seg_start + seg_end) / 2.0
                seg_slide = get_slide(mid)
                new_row = row.copy()
                new_row['start_time'] = seg_start
                new_row['end_time'] = seg_end
                new_row['duration'] = seg_end - seg_start
                new_row['slide'] = seg_slide
                new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

def split_by_slides(blink_df, gaze_df, user_inputs_df):
    """
    Splits the blink and gaze logs by slide based on the user inputs.
    Both DataFrames must have 'start_time' and 'end_time' columns.
    """
    #print("blink_df loaded from:", blink_log_path)
    #print(blink_df.head())
    #print(blink_df.columns)

    slide_df = user_inputs_df[user_inputs_df['action'] == 'slide_transition']
    slide_times = sorted(slide_df['time'].values)

    blink_df = split_events_by_slide(blink_df, slide_times)
    gaze_df = split_events_by_slide(gaze_df, slide_times)

    print(gaze_df)

    return blink_df, gaze_df

    #print(slide_times)


    '''
    blink_df['slide'] = 0
    gaze_df['slide'] = 0

    #print(len(slide_times) - 1)
    for i in range(len(slide_times) - 1):
        start_time = slide_times[i]
        end_time = slide_times[i + 1]
        
        # Assign slide number to blink_df and gaze_df
        blink_df.loc[(blink_df['start_time'] >= start_time) & (blink_df['end_time'] < end_time), 'slide'] = i + 1
        gaze_df.loc[(gaze_df['start_time'] >= start_time) & (gaze_df['end_time'] < end_time), 'slide'] = i + 1

    # Handle the last slide for both DataFrames
    #print([gaze_df['start_time'] >= slide_times[-1]])
    blink_df.loc[(blink_df['start_time'] >= slide_times[-1]) | (blink_df['end_time'] >= slide_times[-1]), 'slide'] = len(slide_times)
    gaze_df.loc[(gaze_df['start_time'] >= slide_times[-1]) | (gaze_df['end_time'] >= slide_times[-1]), 'slide'] = len(slide_times)
    
    #print(blink_df)
    
    return blink_df, gaze_df'''
    
def process_data(video_file, user_inputs_file, output_folder, EAR, full_only=False, shape_predictor="shape_predictor_68_face_landmarks.dat"):
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    run_blink_gaze_tracker(video_file, user_inputs_file, output_folder, base_name, EAR, shape_predictor)

    #if not full_only:
    #    process_split_logs(video_file, user_inputs_file, output_folder)


def cross_check_mind_wandering(user_inputs_df, mind_wandering_df):
    """
    Cross-checks the mind wandering flags with user inputs.
    """
    # Extract the mind wandering actions from user inputs
    # Filter for user-reported mind wandering events.
    mw_user_report = user_inputs_df[user_inputs_df['action'] == 'mind_wandering']
    
    #print(mw_user_report)
    
    # Filter the computed report for slides flagged as mind-wandering.
    mw_computed_report = mind_wandering_df[mind_wandering_df['mind_wandering'] == True].copy()
    #print(mw_computed_report)

    results = []
    # Iterate over each user-reported event.
    for _, user_row in mw_user_report.iterrows():
        user_time = user_row['time']
        # Find computed entries where the user_time is within the time period.
        mw_computed_report['time_start'] = mw_computed_report['time_period'].apply(lambda x: x[0])
        mw_computed_report['time_end'] = mw_computed_report['time_period'].apply(lambda x: x[1])

        #print(mw_computed_report)
        matches = mw_computed_report[
            (mw_computed_report['time_start'] <= user_time) & (mw_computed_report['time_end'] >= user_time)
        ]
        # If one or more matching computed periods are found:
        for _, comp_row in matches.iterrows():
            results.append({
                'user_time': user_time,
                'computed_time_start': comp_row['time_start'],
                'computed_time_end': comp_row['time_end'],
                'slide': comp_row['slide']
            })
    return results
    
    
from slide import Slides

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process blink and gaze logs. Either generate logs from video or (if logs already exist) split them by slide."
    )
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder with video files and user inputs")
    parser.add_argument("--full-only", action="store_true", help="Only generate full blink/gaze logs for the entire presentation; do not split into slides")
    parser.add_argument("--shape_predictor", type=str, default="shape_predictor_68_face_landmarks.dat", help="Path to shape predictor file")
    
    args = parser.parse_args()
    
    word_count = [42,45,0,56,63,68,100,113,100]

    # Loop through video files that follow the naming convention.
    for file_name in os.listdir(args.folder):
        if file_name.startswith("video_recording_") and file_name.endswith(".mp4"):
            subject = file_name[len("video_recording_"):-len(".mp4")]
            video_path = os.path.join(args.folder, file_name)
            user_inputs_file = os.path.join(args.folder, f"user_inputs_{subject}.csv")
            output_folder = os.path.join(args.folder, "processed_data")
            calibration_file = os.path.join(args.folder, f"calibration_values_{subject}.csv")

            calib = pd.read_csv(calibration_file)
            
            EAR = calib['EAR'].values[0]
            velocity = calib['velocity'].values[0]


            
            if os.path.exists(user_inputs_file):
                print(f"Processing subject {subject}...")
                
                blink_log_path = os.path.join(output_folder, f"video_recording_{subject}_blink_log.csv")
                gaze_log_path = os.path.join(output_folder, f"video_recording_{subject}_gaze_log.csv")
                
                process_data(video_path, user_inputs_file, output_folder, EAR,full_only=args.full_only, shape_predictor=args.shape_predictor)
                
                #load_data(blink_log_path, gaze_log_path, user_inputs_file)

                slides = Slides(blink_log_path, gaze_log_path, user_inputs_file, word_count, velocity)
                
                mind_wandering_df = slides.mind_wandering_report()
                #print(mind_wandering_df)

                results = cross_check_mind_wandering(pd.read_csv(user_inputs_file), mind_wandering_df)
                
                print(f"Cross-check results for subject {subject}:")
                print(len(results))


                
