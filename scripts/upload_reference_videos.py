#!/usr/bin/env python3
import boto3
import argparse
import os
import cv2
import numpy as np
import tempfile
import json
import sys
import traceback
import time

# Add the parent directory to the Python path to import from app.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'battingai-opencv-lambda'))

# Import functions from app.py
try:
    from app import extract_frames, save_frames_to_s3, detect_swing_phase
except ImportError as e:
    print(f"Error importing functions from app.py: {e}")
    traceback.print_exc()
    sys.exit(1)

def detect_objects_with_background_subtraction(frames):
    """
    Detect moving objects using background subtraction and contour analysis
    Returns bat and ball trajectories
    """
    # Background subtractor with shadow detection
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    # Store object trajectories
    trajectories = []
    bat_trajectory = []
    ball_trajectory = []
    
    for frame in frames:
        # Apply background subtraction
        fg_mask = backSub.apply(frame)
        
        # Remove shadows and noise
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                # Classify objects based on size and shape
                if 0.8 < aspect_ratio < 1.2 and area < 500:  # Ball-like object
                    ball_trajectory.append((x + w//2, y + h//2))
                elif w > 50 and h < w:  # Bat-like object
                    bat_trajectory.append((x + w//2, y + h//2))
                
                frame_objects.append({
                    'type': 'ball' if 0.8 < aspect_ratio < 1.2 and area < 500 else 'bat',
                    'position': (x + w//2, y + h//2),
                    'size': (w, h),
                    'area': area
                })
        
        trajectories.append(frame_objects)
    
    return trajectories, bat_trajectory, ball_trajectory

def analyze_swing_mechanics(frames):
    """
    Analyze swing mechanics using optical flow and trajectory analysis
    Returns key swing events and their frame indices
    """
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Convert frames to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    # Initialize points to track
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
    if p0 is None:
        return None
        
    # Track points through frames
    trajectories = []
    for i in range(len(gray_frames)-1):
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            gray_frames[i],
            gray_frames[i+1],
            p0,
            None,
            **lk_params
        )
        
        if p1 is not None:
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            if len(good_new) > 0 and len(good_old) > 0:
                # Calculate motion vectors
                motion_vectors_x = good_new[:, 0] - good_old[:, 0]
                motion_vectors_y = good_new[:, 1] - good_old[:, 1]
                
                # Calculate magnitudes and directions
                magnitudes = np.sqrt(motion_vectors_x**2 + motion_vectors_y**2)
                directions = np.arctan2(motion_vectors_y, motion_vectors_x)
                
                # Calculate averages
                avg_magnitude = np.mean(magnitudes)
                avg_direction = np.mean(directions)
                
                trajectories.append({
                    'frame_idx': i,
                    'magnitude': float(avg_magnitude),
                    'direction': float(avg_direction),
                    'points': good_new
                })
            
            # Update points
            p0 = good_new.reshape(-1, 1, 2)
    
    return trajectories

def detect_swing_phase(frames):
    """Detect swing phases using advanced motion analysis and object tracking"""
    if len(frames) < 5:
        return list(range(min(5, len(frames))))
    
    print("Analyzing swing mechanics...")
    
    # Get object trajectories using the same method as user video analysis
    obj_trajectories, bat_trajectory, ball_trajectory = detect_objects_with_background_subtraction(frames)
    
    # Analyze swing mechanics using optical flow
    motion_analysis = analyze_swing_mechanics(frames)
    
    if not motion_analysis:
        print("Failed to analyze motion, falling back to basic detection")
        return [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    
    # Calculate total frames and minimum frame spacing
    total_frames = len(frames)
    min_spacing = max(3, total_frames // 15)  # Ensure at least 3 frames between phases
    
    # First pass: Find contact frame using motion analysis
    contact_idx = None
    max_magnitude = 0
    
    for i, traj in enumerate(motion_analysis):
        # Skip first and last few frames
        if i < min_spacing or i > total_frames - min_spacing:
            continue
        if traj['magnitude'] > max_magnitude:
            max_magnitude = traj['magnitude']
            contact_idx = traj['frame_idx']
    
    if contact_idx is None or contact_idx >= total_frames - min_spacing:
        contact_idx = total_frames // 2
    
    # Second pass: Refine contact frame using ball trajectory changes
    if ball_trajectory:
        ball_velocities = []
        for i in range(1, len(ball_trajectory)):
            dx = ball_trajectory[i][0] - ball_trajectory[i-1][0]
            dy = ball_trajectory[i][1] - ball_trajectory[i-1][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            ball_velocities.append((i-1, velocity))
        
        # Look for sudden changes in ball velocity
        for i in range(1, len(ball_velocities)):
            vel_change = ball_velocities[i][1] - ball_velocities[i-1][1]
            if vel_change > np.mean([v[1] for v in ball_velocities]) * 1.5:
                # Found significant velocity change, this is likely the actual contact
                contact_idx = ball_velocities[i][0]
                break
    
    # Calculate other phases relative to contact
    setup_idx = 0
    load_idx = max(setup_idx + min_spacing, contact_idx - 3 * min_spacing)
    swing_idx = max(load_idx + min_spacing, contact_idx - 2 * min_spacing)
    followthrough_idx = min(total_frames - 1, contact_idx + 2 * min_spacing)
    
    # Combine phases
    phases = [setup_idx, load_idx, swing_idx, contact_idx, followthrough_idx]
    
    # Validate and adjust spacing
    for i in range(1, len(phases)):
        if phases[i] <= phases[i-1] + min_spacing:
            # Try to push the current phase forward
            phases[i] = min(total_frames - 1, phases[i-1] + min_spacing)
    
    # If phases are too clustered at the end, redistribute them
    if phases[-1] >= total_frames - min_spacing:
        frame_step = (total_frames - 1) // 4
        phases = [
            0,
            frame_step,
            frame_step * 2,
            frame_step * 3,
            total_frames - 1
        ]
    
    print(f"Selected frames at: {phases}")
    return phases

def extract_frames(video_path, num_frames=5):
    """Extract key frames from a video file based on swing motion analysis"""
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame sampling rate - aim to process about 100 frames
    frame_step = max(1, total_frames // 100)
    
    # Read frames with initial sampling
    all_frames = []
    frame_indices = []  # Keep track of original frame indices
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every nth frame initially
        if count % frame_step == 0:
            frame = cv2.resize(frame, (640, 360))
            all_frames.append(frame)
            frame_indices.append(count)
            
        count += 1
    
    cap.release()
    
    if len(all_frames) < 5:
        print("Not enough frames, using regular interval sampling")
        return all_frames[:5]
    
    print(f"Processing {len(all_frames)} sampled frames for motion analysis...")
    
    # Analyze motion in the sampled frames
    motion_scores = []
    prev_frame = None
    
    for frame in all_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            motion_scores.append(magnitude)
        prev_frame = gray
    
    # Find significant motion changes
    motion_changes = []
    for i in range(1, len(motion_scores)):
        change = motion_scores[i] - motion_scores[i-1]
        motion_changes.append((i, change))
    
    # Sort by absolute magnitude of change
    motion_changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Find potential key moments
    setup_candidates = motion_changes[-len(motion_changes)//3:]  # Low motion
    contact_candidates = motion_changes[:len(motion_changes)//3]  # High motion
    
    # Select key frames
    setup_idx = 0  # Always start with first frame
    contact_idx = None
    
    # Find contact frame (highest motion spike)
    for idx, _ in contact_candidates:
        if idx > len(all_frames) // 3 and idx < 2 * len(all_frames) // 3:  # Contact should be in middle third
            contact_idx = idx
            break
    
    if contact_idx is None:
        contact_idx = len(all_frames) // 2
    
    # Calculate other phases based on contact
    load_idx = max(setup_idx + 1, contact_idx - len(all_frames) // 4)
    swing_idx = max(load_idx + 1, contact_idx - 2)
    followthrough_idx = min(len(all_frames) - 1, contact_idx + len(all_frames) // 6)
    
    # Ensure proper spacing
    min_spacing = len(all_frames) // 10
    phases = [setup_idx, load_idx, swing_idx, contact_idx, followthrough_idx]
    
    # Adjust spacing if needed
    for i in range(1, len(phases)):
        if phases[i] <= phases[i-1] + min_spacing:
            phases[i] = min(len(all_frames) - 1, phases[i-1] + min_spacing)
    
    # If frames are too clustered, redistribute
    if phases[-1] - phases[0] < len(all_frames) // 2:
        total_span = len(all_frames) - 1
        phases = [
            0,                          # Setup
            total_span // 4,            # Load
            total_span // 2,            # Swing
            (3 * total_span) // 4,      # Contact
            total_span                  # Follow-through
        ]
    
    # Map back to original frame indices
    final_indices = [frame_indices[i] for i in phases]
    print(f"Selected original frame indices: {final_indices}")
    
    # Reopen video to extract the exact frames we want
    cap = cv2.VideoCapture(video_path)
    final_frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in final_indices:
            frame = cv2.resize(frame, (640, 360))
            final_frames.append(frame)
            
        frame_count += 1
        if len(final_frames) == len(final_indices):
            break
            
    cap.release()
    
    return final_frames

def upload_reference_video(s3_client, bucket_name, video_path, player_id, player_name):
    """Upload a reference video and its frames to S3 using the same processing as user videos"""
    # Upload the video
    video_key = f"reference/{player_id}.mp4"
    print(f"Uploading {video_path} to s3://{bucket_name}/{video_key}")
    s3_client.upload_file(video_path, bucket_name, video_key)
    
    # Create a reference ID similar to analysis_id for user videos
    reference_id = f"{player_id}_{int(time.time())}"
    
    # Extract frames using the same function as user videos
    print(f"Extracting frames from {video_path} using app.py extract_frames function")
    try:
        # Download video to a temporary file if it's an S3 URL
        if video_path.startswith('s3://'):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_path = temp_video.name
                s3_path = video_path[5:].split('/', 1)
                s3_bucket = s3_path[0]
                s3_key = s3_path[1]
                s3_client.download_file(s3_bucket, s3_key, temp_path)
                frames = extract_frames(temp_path)
                os.unlink(temp_path)
        else:
            frames = extract_frames(video_path)
        
        print(f"Successfully extracted {len(frames)} frames")
        
        # Save frames to S3 using the same function as user videos
        frame_paths, frame_urls = save_frames_to_s3(frames, f"reference/{player_id}")
        
        # Create metadata similar to user video metadata
        metadata = {
            'player_id': player_id,
            'player_name': player_name,
            'video_key': video_key,
            'frame_paths': frame_paths,
            'frame_urls': frame_urls,
            'frame_count': len(frames),
            'created_at': int(time.time())
        }
        
        # Upload metadata
        metadata_key = f"reference/{player_id}/metadata.json"
        print(f"Uploading metadata to s3://{bucket_name}/{metadata_key}")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        print(f"Successfully uploaded reference video for {player_name}")
        return True
    except Exception as e:
        print(f"Error processing reference video: {str(e)}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload reference MLB player videos to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--player-id', required=True, help='Player ID (e.g., bryce_harper)')
    parser.add_argument('--player-name', required=True, help='Player name (e.g., Bryce Harper)')
    parser.add_argument('--profile', help='AWS profile name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    # Create S3 client
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3_client = session.client('s3')
    
    # Upload reference video
    success = upload_reference_video(s3_client, args.bucket, args.video, args.player_id, args.player_name)
    
    if not success:
        print(f"Failed to upload reference video for {args.player_name}")
        sys.exit(1)
    else:
        print(f"Successfully processed reference video for {args.player_name}")
        sys.exit(0)

if __name__ == '__main__':
    main()