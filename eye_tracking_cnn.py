from psychopy import visual, core, event, monitors
from cnn_eye_tracking import EyeCNN
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib 
from sklearn.model_selection import train_test_split

# ======================
# SETUP
# ======================
# Monitor and window setup
mon = monitors.Monitor('myMonitor', width=120.0, distance=100.0)
mon.setSizePix((1536, 864))
win = visual.Window(size=(1536, 864), 
                    monitor=mon, 
                    fullscr=True,
                    screen=0,
                    color=(0.8, 0.8, 0.8), 
                    units='pix')



# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye tracking parameters
EAR_THRESHOLD = 0.25
FIXATION_FRAMES = 3  # ~100ms at 30fps
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ======================
# STIMULI (3 TASKS)
# ======================
def create_task1():
    """Syllables reading task (9x10 matrix)"""
    syllables = [
        ["ba", "no", "si", "me", "fl", "zr", "kle", "pra", "brik", "flan"],
        ["pe", "le", "ko", "so", "tr", "sk", "ple", "sle", "plot", "skop"],
        ["bu", "du", "ha", "ka", "gl", "fr", "ble", "gre", "blok", "fram"],
        ["ve", "ze", "lo", "po", "ch", "kl", "vre", "kle", "vres", "klon"],
        ["fi", "ri", "mi", "ni", "pl", "dr", "pre", "dre", "ples", "drif"],
        ["ji", "ci", "ho", "jo", "sm", "sn", "sne", "sme", "smet", "snop"],
        ["ku", "lu", "pa", "ra", "br", "gr", "dru", "bru", "drum", "grum"],
        ["xi", "yi", "wo", "zo", "cr", "gl", "cro", "glo", "crot", "glos"],
        ["wu", "we", "bi", "di", "sp", "st", "spo", "sto", "spol", "stol"]
    ]
    
    stimuli = []
    y_pos = 300
    for row in syllables:
        x_pos = -700
        for syllable in row:
            stim = visual.TextStim(win, text=syllable, pos=(x_pos, y_pos), height=40, color='black')
            stimuli.append({'stim': stim, 'roi': (x_pos-50, y_pos-25, x_pos+50, y_pos+25)})
            x_pos += 150
        y_pos -= 80
    
    # Fixation cross for ending
    fixation = visual.TextStim(win, text='+', pos=(800, -400), height=40, color='black')
    return stimuli, fixation

def create_task2():
    """Meaningful text reading task"""
    story = [
        "Tom looked out the window and saw a squirrel.", 
        "The squirrel jumped from branch to branch,",
        "quick and nimble. Tom wished he could ",   
        "climb trees like that. But the squirrel ",  
        "was gathering nuts for winter, and Tom ",  
        "had to go to school. His backpack ",  
        "waited by the door, heavy with books." 
    ]
    
    stimuli = []
    y_pos = 300
    for line in story:
        stim = visual.TextStim(win, text=line, pos=(0, y_pos), height=40, color='black')
        stimuli.append({'stim': stim, 'roi': (-800, y_pos-25, 800, y_pos+25)})
        y_pos -= 80
    
    fixation = visual.TextStim(win, text='+', pos=(800, -400), height=40, color='black')
    return stimuli, fixation

def create_task3():
    """Pseudo text reading task"""
    pseudo_text = [
        "The flibber jabbled near the zorknox.",
        "Wombly squigs trempled in the quagspoon.",  
        "Ploovitz narfed the glipswitch when",
        "the brizzles frapped. Zangle doofers",
        "wibbled the mungo with a snorfle.",
        "Glimpy vortens kexed the plumbot",
        "but the dweezils qued in reply."
    ]
    
    stimuli = []
    y_pos = 300
    for line in pseudo_text:
        stim = visual.TextStim(win, text=line, pos=(0, y_pos), height=40, color='black')
        stimuli.append({'stim': stim, 'roi': (-800, y_pos-25, 800, y_pos+25)})
        y_pos -= 80
    
    fixation = visual.TextStim(win, text='+', pos=(800, -400), height=40, color='black')
    return stimuli, fixation

# ======================
# EYE TRACKING FUNCTIONS
# ======================
def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def is_in_roi(gaze_pos, roi):
    """Check if gaze is within a region of interest"""
    x, y = gaze_pos
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]

def process_frame(frame, trial_data, current_task):
    """Process each frame for eye tracking metrics"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye coordinates
        left_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE])
        right_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
        
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Gaze center
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        gaze_pos = ((left_center + right_center) / 2).astype(int)
        
        # ROI tracking for current task
        for roi in current_task['rois']:
            if is_in_roi(gaze_pos, roi['coords']):
                roi['fixations'] += 1
                roi['duration'] += 1/60.0  # Assuming 60Hz
        
        return gaze_pos, ear
    return None, None

# ======================
# EXPERIMENT FLOW
# ======================
def run_experiment():
    # Initialize data storage
    all_data = {
        'task1': {'fixations': [], 'saccades': [], 'rois': []},
        'task2': {'fixations': [], 'saccades': [], 'rois': []},
        'task3': {'fixations': [], 'saccades': [], 'rois': []}
    }
    
    # Create tasks
    task1_stim, task1_fix = create_task1()
    task2_stim, task2_fix = create_task2()
    task3_stim, task3_fix = create_task3()
    
    # Webcam setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found.")
        return
    
    # Run all three tasks
    for task_idx, (task_stim, task_fix, task_name) in enumerate([
        (task1_stim, task1_fix, 'task1'),
        (task2_stim, task2_fix, 'task2'),
        (task3_stim, task3_fix, 'task3')
    ], 1):
        print(f"\n=== Starting {task_name} ===")
        
        # Initialize task-specific trackers
        fixation_durations = []
        saccade_counts = 0
        frame_counter = 0
        fixation_start = None
        prev_gaze_pos = None
        fixation_positions = []
        saccades = []
        
        # Create ROIs for this task
        rois = [{'name': f'roi_{i}', 'coords': s['roi'], 'fixations': 0, 'duration': 0} 
                for i, s in enumerate(task_stim)]
        all_data[task_name]['rois'] = rois
        
        # Show instructions (updated)
        instr_text = f"Task {task_idx}: Read the following {'syllables' if task_idx==1 else 'text'} aloud\n\nPress SPACE when ready to begin\n\nPress SPACE again when finished reading"
        instr = visual.TextStim(win, text=instr_text, height=40, color='black')
        instr.draw()
        win.flip()
        event.waitKeys(keyList=['space'])  # Wait for space to begin
        
        # Present stimuli
        for stim in task_stim:
            stim['stim'].draw()
        win.flip()
        
        # Eye tracking during task
        start_time = datetime.now()
        task_complete = False
        
        while not task_complete:
            # Check for space bar press to end task
            keys = event.getKeys()
            if 'space' in keys:
                task_complete = True
            if 'q' in keys:  # Still allow emergency quit
                cap.release()
                win.close()
                return
                
            # Process webcam frame
            ret, frame = cap.read()
            if not ret:
                break
                
            gaze_pos, ear = process_frame(frame, all_data[task_name], 
                                        {'rois': rois, 'name': task_name})
            
            if gaze_pos is not None:
                # Fixation detection logic (unchanged)
                if ear > EAR_THRESHOLD:
                    if fixation_start is None:
                        fixation_start = frame_counter
                    elif prev_gaze_pos is not None:
                        dist_px = np.linalg.norm(gaze_pos - prev_gaze_pos)
                        if dist_px > 10:  # Saccade threshold
                            saccade_counts += 1
                            saccades.append({
                                'start': prev_gaze_pos.tolist(),
                                'end': gaze_pos.tolist(),
                                'start_frame': fixation_start,
                                'end_frame': frame_counter
                            })
                            fixation_start = frame_counter
                    fixation_positions.append(gaze_pos)
                    prev_gaze_pos = gaze_pos
                else:
                    if fixation_start is not None:
                        duration = frame_counter - fixation_start
                        if duration >= FIXATION_FRAMES:
                            fixation_durations.append(duration / 60.0)
                            all_data[task_name]['fixations'].append({
                                'start_frame': fixation_start,
                                'end_frame': frame_counter,
                                'duration': duration / 60.0,
                                'x': fixation_positions[-1][0],
                                'y': fixation_positions[-1][1]
                            })
                    fixation_start = None
                    fixation_positions = []
            
            frame_counter += 1
        
        # Save task metrics (unchanged)
        all_data[task_name]['saccades'] = saccades
        all_data[task_name]['total_fixations'] = len(fixation_durations)
        all_data[task_name]['mean_fixation_duration'] = np.mean(fixation_durations) if fixation_durations else 0
        all_data[task_name]['total_saccades'] = saccade_counts
        
        # Show task completion
        complete = visual.TextStim(win, text="Task complete!\n\nPress SPACE to continue", 
                                 height=40, color='black')
        complete.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
    
    # Cleanup and save data (unchanged)
    cap.release()
    win.close()
    save_data(all_data)
    analyze_dyslexia(all_data)

def save_data(all_data):
    """Save all collected metrics in the specified format"""
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, data in all_data.items():
        # Create participant and trial identifiers
        sid = 1  # Would normally come from participant info
        trialid = {'task1': 1, 'task2': 2, 'task3': 3}[task_name]
        
        # ==============================================
        # 1. Fixations File
        # ==============================================
        fixations = []
        for i, fix in enumerate(data.get('fixations', [])):
            fixations.append({
                'id': i + 1,
                'sid': sid,
                'trialid': trialid,
                'start_ms': fix.get('start_frame', 0) * 16.667,
                'end_ms': fix.get('end_frame', 0) * 16.667,
                'duration_ms': fix.get('duration', 0) * 1000,
                'fix_x': fix.get('x', 0),
                'fix_y': fix.get('y', 0),
                'orig_fix_x': fix.get('x', 0),
                'orig_fix_y': fix.get('y', 0),
                'disp_x': 0,
                'disp_y': 0
            })
        
        if fixations:  # Only save if we have data
            fix_df = pd.DataFrame(fixations)
            fix_df.to_csv(
                f"output/{timestamp}_{task_name}_fixations.csv", 
                index=False
            )
        
        # ==============================================
        # 2. Saccades File
        # ==============================================
        saccades = []
        for i, sacc in enumerate(data.get('saccades', [])):
            duration_frames = sacc.get('end_frame', 0) - sacc.get('start_frame', 0)
            duration_ms = duration_frames * 16.667
            start_pos = sacc.get('start', [0, 0])
            end_pos = sacc.get('end', [0, 0])
            distance = dist.euclidean(start_pos, end_pos)
            
            saccades.append({
                'id': i + 1,
                'sid': sid,
                'trialid': trialid,
                'start_ms': sacc.get('start_frame', 0) * 16.667,
                'end_ms': sacc.get('end_frame', 0) * 16.667,
                'duration_ms': duration_ms,
                'start_x': start_pos[0],
                'start_y': start_pos[1],
                'end_x': end_pos[0],
                'end_y': end_pos[1],
                'ampl_x': end_pos[0] - start_pos[0],
                'ampl_y': end_pos[1] - start_pos[1],
                'avg_vel_x': (end_pos[0] - start_pos[0]) / duration_ms if duration_ms > 0 else 0,
                'avg_vel_y': (end_pos[1] - start_pos[1]) / duration_ms if duration_ms > 0 else 0,
                'peak_vel_x': 0,  # Placeholder - would need frame-by-frame data
                'peak_vel_y': 0,   # Placeholder
                'avg_vel': distance / duration_ms if duration_ms > 0 else 0,
                'peak_vel': 0,     # Placeholder
                'ampl': distance
            })
        
        if saccades:  # Only save if we have data
            sacc_df = pd.DataFrame(saccades)
            # Ensure all required columns exist
            required_columns = [
                'id', 'sid', 'trialid', 'start_ms', 'end_ms', 'duration_ms',
                'start_x', 'start_y', 'end_x', 'end_y', 'ampl_x', 'ampl_y',
                'avg_vel_x', 'avg_vel_y', 'peak_vel_x', 'peak_vel_y',
                'avg_vel', 'peak_vel', 'ampl'
            ]
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in sacc_df.columns:
                    sacc_df[col] = 0
            
            sacc_df.to_csv(
                f"output/{timestamp}_{task_name}_saccades.csv", 
                index=False,
                columns=required_columns
            ) 
        
        # ==============================================
        # 3. Metrics File
        # ==============================================
        # Trial-level metrics
        n_fix_trial = len(data['fixations'])
        sum_fix_dur_trial = sum(f['duration'] for f in data['fixations']) * 1000  # ms
        dwell_time_trial = sum_fix_dur_trial
        mean_fix_dur_trial = sum_fix_dur_trial / n_fix_trial if n_fix_trial > 0 else 0
        
        n_sacc_trial = len(data['saccades'])
        sum_sacc_dur_trial = sum((s['end_frame']-s['start_frame'])*16.667 for s in data['saccades'])
        mean_sacc_dur_trial = sum_sacc_dur_trial / n_sacc_trial if n_sacc_trial > 0 else 0
        mean_sacc_ampl_trial = np.mean([dist.euclidean(s['start'], s['end']) for s in data['saccades']]) if data['saccades'] else 0
        
        # Saccade direction analysis
        regress = [s for s in data['saccades'] if s['end'][0] < s['start'][0]]
        progress = [s for s in data['saccades'] if s['end'][0] >= s['start'][0]]
        n_regress_trial = len(regress)
        n_progress_trial = len(progress)
        ratio_progress_regress_trial = n_progress_trial / max(1, n_regress_trial)
        
        # AOI (ROI) metrics
        aoi_metrics = []
        for i, roi in enumerate(data['rois']):
            n_fix_aoi = roi['fixations']
            sum_fix_dur_aoi = roi['duration'] * 1000  # ms
            mean_fix_dur_aoi = sum_fix_dur_aoi / max(1, n_fix_aoi)
            skipped_aoi = 1 if n_fix_aoi == 0 else 0
            
            # For first pass metrics - would need more sophisticated tracking
            first_fix = next((f for f in data['fixations'] if is_in_roi([f['x'], f['y']], roi['coords'])), None)
            
            aoi_metrics.append({
                'sid': sid,
                'trialid': trialid,
                'aoi_id': i + 1,
                'dwell_time_aoi': sum_fix_dur_aoi,
                'n_fix_aoi': n_fix_aoi,
                'sum_fix_dur_aoi': sum_fix_dur_aoi,
                'mean_fix_dur_aoi': mean_fix_dur_aoi,
                'skipped_aoi': skipped_aoi,
                'n_fix_first_visit_aoi': 1 if first_fix else 0,
                'first_fix_dur_aoi': first_fix['duration']*1000 if first_fix else 0,
                'first_fix_land_pos_aoi': first_fix['x'] - roi['coords'][0] if first_fix else 0,
                'dwell_time_first_visit_aoi': first_fix['duration']*1000 if first_fix else 0,
                'sum_fix_dur_first_visit_aoi': first_fix['duration']*1000 if first_fix else 0,
                'sum_fix_dur_after_first_visit_aoi': max(0, sum_fix_dur_aoi - (first_fix['duration']*1000 if first_fix else 0)),
                'dwell_time_rereading_aoi': max(0, sum_fix_dur_aoi - (first_fix['duration']*1000 if first_fix else 0)),
                'n_revisits_aoi': max(0, n_fix_aoi - 1)
            })
        
        # Combine trial and AOI metrics
        metrics = {
            'sid': sid,
            'trialid': trialid,
            'n_fix_trial': n_fix_trial,
            'sum_fix_dur_trial': sum_fix_dur_trial,
            'dwell_time_trial': dwell_time_trial,
            'mean_fix_dur_trial': mean_fix_dur_trial,
            'n_sacc_trial': n_sacc_trial,
            'sum_sacc_dur_trial': sum_sacc_dur_trial,
            'mean_sacc_dur_trial': mean_sacc_dur_trial,
            'mean_sacc_ampl_trial': mean_sacc_ampl_trial,
            'ratio_progress_regress_trial': ratio_progress_regress_trial,
            'n_between_line_regress_trial': 0,  # Would need line tracking
            'n_within_line_regress_trial': 0,   # Would need line tracking
            'n_regress_trial': n_regress_trial,
            'n_progress_trial': n_progress_trial,
            'n_transit_trial': 0  # Would need more sophisticated tracking
        }
        
        # Save trial metrics
        pd.DataFrame([metrics]).to_csv(
            f"output/{timestamp}_{task_name}_metrics_trial.csv",
            index=False
        )
        
        # Save AOI metrics
        pd.DataFrame(aoi_metrics).to_csv(
            f"output/{timestamp}_{task_name}_metrics_aoi.csv",
            index=False
        )
def analyze_dyslexia(all_data):
    """Predict dyslexia using saved model"""
    model_path = "cnn_eye_tracking.pth"
    if not os.path.exists(model_path):
        print("[WARNING] No trained model found. Skipping prediction.")
        return
    
    # Prepare features in the exact format expected by the model
    features = []
    
    # For each task, extract and structure the features
    for task_name in ['task1', 'task2', 'task3']:
        data = all_data[task_name]
        
        # ==============================================
        # 1. Fixation Features
        # ==============================================
        fix_features = {
            'duration_ms': np.mean([f['duration']*1000 for f in data['fixations']]) if data['fixations'] else 0,
            'fix_x': np.mean([f['x'] for f in data['fixations']]) if data['fixations'] else 0,
            'fix_y': np.mean([f['y'] for f in data['fixations']]) if data['fixations'] else 0,
            'disp_x': 0,  # Not currently tracked
            'disp_y': 0   # Not currently tracked
        }
        
        # ==============================================
        # 2. Saccade Features
        # ==============================================
        sacc_features = {
            'duration_ms': np.mean([(s['end_frame']-s['start_frame'])*16.667 for s in data['saccades']]) if data['saccades'] else 0,
            'ampl': np.mean([dist.euclidean(s['start'], s['end']) for s in data['saccades']]) if data['saccades'] else 0,
            'avg_vel': np.mean([dist.euclidean(s['start'], s['end'])/((s['end_frame']-s['start_frame'])*16.667) 
                        if (s['end_frame']-s['start_frame']) > 0 else 0 
                        for s in data['saccades']]) if data['saccades'] else 0,
            'n_regress': len([s for s in data['saccades'] if s['end'][0] < s['start'][0]]),
            'n_progress': len([s for s in data['saccades'] if s['end'][0] >= s['start'][0]])
        }
        
        # ==============================================
        # 3. Metrics Features
        # ==============================================
        n_fix = len(data['fixations'])
        sum_fix_dur = sum(f['duration'] for f in data['fixations']) if data['fixations'] else 0
        mean_fix_dur = sum_fix_dur / n_fix if n_fix > 0 else 0
        
        met_features = {
            'n_fix_trial': n_fix,
            'sum_fix_dur_trial': sum_fix_dur * 1000,  # Convert to ms
            'mean_fix_dur_trial': mean_fix_dur * 1000,  # Convert to ms
            'n_sacc_trial': len(data['saccades']),
            'ratio_progress_regress': sacc_features['n_progress'] / max(1, sacc_features['n_regress']),
            'n_regress_trial': sacc_features['n_regress']
        }
        
        # Combine all features for this task
        task_features = list(fix_features.values()) + list(sacc_features.values()) + list(met_features.values())
        features.extend(task_features)
    
    # Convert to numpy array and reshape for the model
    features = np.array(features).astype(np.float32)
    
    # Pad and reshape to match model expectations (10xN)
    num_features = len(features)
    height = 10
    width = int(np.ceil(num_features / height))
    padded = np.zeros(height * width)
    padded[:num_features] = features
    features = padded.reshape(1, 1, height, width)  # Add batch and channel dims
    
    # Load and run model
    model = EyeCNN((height, width))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        label = "Dyslexic" if predicted.item() == 1 else "Non-Dyslexic"
    
    print(f"\n[RESULT] Predicted reading pattern: {label}")

# Run the experiment
if __name__ == "__main__":
    run_experiment()