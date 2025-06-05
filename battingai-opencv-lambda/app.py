import json
import boto3
import os
import traceback
import tempfile
import cv2
import numpy as np
import base64  # Add base64 for encoding binary data
import time    # Add time for timestamps
import google.generativeai as genai
import base64

s3_client = boto3.client('s3')
bucket_name = os.environ.get('BUCKET_NAME', 'battingai-videobucket-ayk9m1uehbg2')

# Player name mapping
PLAYER_NAMES = {
    'bryce_harper': 'Bryce Harper',
    'brandon_lowe': 'Brandon Lowe',
    'mike_trout': 'Mike Trout',
    'jonathan_aranda': 'Jonathan Aranda',
    'aaron_judge': 'Aaron Judge',
    'shohei_ohtani': 'Shohei Ohtani'
}

def detect_batter_position(frame):
    """Detect the batter's position in the frame"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the batter)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    
    return None

def align_frame(frame, batter_position=None):
    """Align the frame based on batter position"""
    if batter_position is None:
        batter_position = detect_batter_position(frame)
    
    if batter_position:
        x, y, w, h = batter_position
        # Create a centered frame with the batter in the middle
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate translation to center the batter
        tx = center_x - (x + w // 2)
        ty = center_y - (y + h // 2)
        
        # Create translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        aligned_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))
        return aligned_frame
    
    return frame

def detect_baseball(frame, prev_ball=None, frame_history=None):
    """Detect baseball in the frame using circle detection.
    
    Improved Robustness:
    - Dynamic radius thresholds based on frame size
    - Velocity-based validation using previous ball positions
    - Basic ML-based ball detection using color and shape features
    - Trajectory prediction for better tracking
    """
    # Calculate dynamic radius thresholds based on frame size
    height, width = frame.shape[:2]
    frame_diagonal = np.sqrt(height**2 + width**2)
    min_radius = max(3, int(frame_diagonal * 0.005))  # 0.5% of diagonal
    max_radius = max(25, int(frame_diagonal * 0.02))  # 2% of diagonal
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles with dynamic parameters
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(50, int(width * 0.05)),  # At least 5% of width
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Initialize candidates list
    candidates = []
    
    if circles is not None:
        # Convert circles to list of candidates with confidence scores
        for circle in circles[0]:
            x, y, r = circle
            
            # Calculate confidence based on circle properties
            # Check if circle is within reasonable bounds
            if 0 <= x < width and 0 <= y < height:
                # Extract region around the circle
                x1, y1 = max(0, int(x - r)), max(0, int(y - r))
                x2, y2 = min(width, int(x + r)), min(height, int(y + r))
                
                if x1 < x2 and y1 < y2:  # Valid region
                    roi = frame[y1:y2, x1:x2]
                    
                    # Calculate color features (baseballs are typically white)
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # Check for white/light color (high V, low S)
                        avg_h = np.mean(hsv_roi[:,:,0])
                        avg_s = np.mean(hsv_roi[:,:,1])
                        avg_v = np.mean(hsv_roi[:,:,2])
                        
                        # Baseball color confidence (higher for white/light objects)
                        color_conf = min(1.0, (avg_v / 255.0) * (1.0 - avg_s / 255.0))
                        
                        # Shape confidence (higher for circular objects)
                        # Use the HoughCircles confidence which is already factored in
                        shape_conf = 0.7
                        
                        # Combined confidence
                        confidence = 0.6 * color_conf + 0.4 * shape_conf
                        
                        candidates.append({
                            'position': (x, y, r),
                            'confidence': confidence
                        })
    
    # Velocity-based validation if we have previous ball position
    if prev_ball is not None and candidates:
        prev_x, prev_y = prev_ball[0], prev_ball[1]
        
        for candidate in candidates:
            x, y, _ = candidate['position']
            
            # Calculate distance from previous position
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            
            # Adjust confidence based on reasonable movement
            # Too much movement is unlikely for consecutive frames
            max_reasonable_distance = width * 0.2  # 20% of frame width
            
            if distance <= max_reasonable_distance:
                # Higher confidence for reasonable movement
                movement_conf = 1.0 - (distance / max_reasonable_distance)
                candidate['confidence'] *= (0.7 + 0.3 * movement_conf)
            else:
                # Lower confidence for excessive movement
                candidate['confidence'] *= 0.5
    
    # Trajectory prediction if we have frame history
    if frame_history and len(frame_history) >= 2:
        # Simple linear prediction based on last two positions
        last_positions = frame_history[-2:]
        
        if all(pos is not None for pos in last_positions):
            # Calculate predicted position
            x1, y1 = last_positions[0][0], last_positions[0][1]
            x2, y2 = last_positions[1][0], last_positions[1][1]
            
            # Predict next position with simple linear extrapolation
            pred_x = x2 + (x2 - x1)
            pred_y = y2 + (y2 - y1)
            
            # If no candidates found, create one based on prediction
            if not candidates:
                # Use average radius from history
                avg_radius = np.mean([pos[2] for pos in last_positions if len(pos) > 2])
                candidates.append({
                    'position': (pred_x, pred_y, avg_radius),
                    'confidence': 0.5  # Lower confidence for predicted position
                })
            else:
                # Adjust confidence based on proximity to predicted position
                for candidate in candidates:
                    x, y, _ = candidate['position']
                    distance_to_prediction = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                    
                    # Higher confidence for candidates close to prediction
                    if distance_to_prediction < width * 0.1:  # Within 10% of frame width
                        prediction_conf = 1.0 - (distance_to_prediction / (width * 0.1))
                        candidate['confidence'] *= (0.8 + 0.2 * prediction_conf)
    
    # Select best candidate
    if candidates:
        best_candidate = max(candidates, key=lambda c: c['confidence'])
        return best_candidate['position']
    
    return None

def detect_bat(frame, prev_bat=None):
    """Detect bat in the frame using edge detection and line detection.
    
    Enhanced Detection:
    - Adaptive thresholding based on frame lighting
    - Bat angle and speed tracking
    - Bat path prediction
    - Confidence scoring
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate adaptive thresholds based on image brightness
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Adjust Canny thresholds based on image brightness
    low_threshold = max(10, int(mean_brightness * 0.3))
    high_threshold = min(250, int(mean_brightness * 0.8 + std_brightness))
    
    # Apply Canny edge detection with adaptive thresholds
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    
    # Apply morphological operations to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Detect lines with parameters adjusted based on frame size
    height, width = frame.shape[:2]
    min_line_length = max(40, int(width * 0.08))  # At least 8% of frame width
    max_line_gap = max(10, int(width * 0.02))     # At least 2% of frame width
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return None, 0.0
    
    # Process detected lines
    bat_candidates = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line properties
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Normalize angle to 0-180 range
        if angle < 0:
            angle += 180
        
        # Calculate bat-like score based on properties
        # Bats typically have:
        # 1. Significant length
        # 2. Angle typically between 0-60 or 120-180 degrees during swing
        # 3. Position typically in lower part of frame
        
        # Length score (longer is better, up to a point)
        length_score = min(1.0, length / (width * 0.3))
        
        # Angle score (higher for typical bat angles)
        angle_score = 0.0
        if (0 <= angle <= 60) or (120 <= angle <= 180):
            angle_score = 1.0 - min(1.0, abs(angle - 30) / 30) if angle <= 60 else 1.0 - min(1.0, abs(angle - 150) / 30)
        
        # Position score (higher for positions in lower part of frame)
        mid_y = (y1 + y2) / 2
        position_score = min(1.0, mid_y / height)
        
        # Combined score
        combined_score = 0.5 * length_score + 0.3 * angle_score + 0.2 * position_score
        
        bat_candidates.append({
            'line': (x1, y1, x2, y2),
            'length': length,
            'angle': angle,
            'score': combined_score
        })
    
    # If we have previous bat position, use it to improve detection
    if prev_bat is not None:
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_bat
        prev_mid_x = (prev_x1 + prev_x2) / 2
        prev_mid_y = (prev_y1 + prev_y2) / 2
        prev_angle = np.arctan2(prev_y2 - prev_y1, prev_x2 - prev_x1) * 180 / np.pi
        if prev_angle < 0:
            prev_angle += 180
        
        # Adjust scores based on proximity to previous position and angle
        for candidate in bat_candidates:
            x1, y1, x2, y2 = candidate['line']
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Distance to previous midpoint
            distance = np.sqrt((mid_x - prev_mid_x)**2 + (mid_y - prev_mid_y)**2)
            distance_score = 1.0 - min(1.0, distance / (width * 0.2))
            
            # Angle difference
            angle_diff = min(abs(candidate['angle'] - prev_angle), 180 - abs(candidate['angle'] - prev_angle))
            angle_diff_score = 1.0 - min(1.0, angle_diff / 45)  # 45 degrees max difference
            
            # Adjust score based on previous position and angle
            candidate['score'] = 0.6 * candidate['score'] + 0.25 * distance_score + 0.15 * angle_diff_score
    
    # Select best candidate
    if bat_candidates:
        best_candidate = max(bat_candidates, key=lambda x: x['score'])
        return best_candidate['line'], best_candidate['score']
    
    return None, 0.0

def is_contact_frame(frame, prev_frame=None):
    """Detect if this frame shows bat-ball contact"""
    # Try to detect the baseball
    ball = detect_baseball(frame)
    if ball is None:
        return False, 0.0
    
    # Try to detect the bat
    bat, bat_confidence = detect_bat(frame)
    if bat is None:
        return False, 0.0
    
    # Calculate distance between bat and ball
    if bat is not None and ball is not None:
        # Convert bat line to midpoint
        bat_mid_x = (bat[0] + bat[2]) / 2
        bat_mid_y = (bat[1] + bat[3]) / 2
        
        # Calculate distance between bat midpoint and ball center
        distance = np.sqrt(
            (bat_mid_x - ball[0])**2 + 
            (bat_mid_y - ball[1])**2
        )
        
        # If ball is very close to bat, this might be contact
        contact_score = 1.0 - min(1.0, distance / 50.0)  # 50 pixels threshold
        
        # Adjust contact score based on bat confidence
        contact_score *= bat_confidence
        
        # If we have a previous frame, check for sudden changes
        if prev_frame is not None:
            prev_ball = detect_baseball(prev_frame)
            if prev_ball is not None:
                # Calculate ball movement
                ball_movement = np.sqrt(
                    (ball[0] - prev_ball[0])**2 +
                    (ball[1] - prev_ball[1])**2
                )
                # Sudden direction changes indicate contact
                if ball_movement > 10:  # Significant movement
                    contact_score *= 1.5  # Boost the score
        
        return contact_score > 0.7, contact_score
    
    return False, 0.0

def detect_swing_phase(frames, pose_results=None):
    """Detect swing phases using combined optical flow, motion analysis and pose detection.
    
    Args:
        frames: List of video frames to analyze
        pose_results: Optional pre-computed pose data. If None, will compute it using detect_poses_with_gemini
    """
    if len(frames) < 5:
        return list(range(min(5, len(frames))))
    
    print("Analyzing swing mechanics with enhanced detection...")
    
    total_frames = len(frames)
    frame_motions = []
    prev_frame = None
    
    # Use provided pose data or get it if not provided
    if pose_results is None:
        pose_results = detect_poses_with_gemini(frames)
    has_pose_data = pose_results is not None and len(pose_results) > 0
    
    # Use pose data to identify key swing phases
    if has_pose_data:
        key_frames = []
        for i, pose_data in enumerate(pose_results):
            if pose_data is None:
                continue
                
            # Extract key points
            keypoints = pose_data.get('keypoints', {})
            bat_data = pose_data.get('bat', {})
            
            # Calculate phase indicators
            if i == 0:
                # Setup phase - first frame
                key_frames.append(i)
            else:
                prev_pose = pose_results[i-1]
                if prev_pose:
                    # Check if required keypoints exist before calculating movement
                    required_keypoints = ['left_hip', 'right_hip']
                    has_required_keypoints = all(k in keypoints for k in required_keypoints) and all(k in prev_pose['keypoints'] for k in required_keypoints)
                    
                    hip_movement = 0
                    if has_required_keypoints:
                        # Load phase - check for significant hip rotation and weight shift
                        prev_hips = (prev_pose['keypoints']['left_hip'], prev_pose['keypoints']['right_hip'])
                        curr_hips = (keypoints['left_hip'], keypoints['right_hip'])
                        hip_movement = calculate_point_movement(prev_hips, curr_hips)
                    
                    # Swing phase - check for bat movement and arm extension
                    prev_bat = prev_pose.get('bat', {}).get('grip_position', {})
                    curr_bat = bat_data.get('grip_position', {})
                    bat_movement = calculate_point_movement([prev_bat], [curr_bat])
                    
                    # Add key frame if significant movement detected or if this is a key position
                    if hip_movement > 0.1 or bat_movement > 0.15 or i == len(pose_results)//2 or i == len(pose_results)-1:
                        key_frames.append(i)
        
        # Ensure we have enough key frames
        if len(key_frames) < 5:
            # Add intermediate frames to get to 5 key phases
            needed_frames = 5 - len(key_frames)
            frame_indices = np.linspace(0, total_frames - 1, needed_frames + 2)[1:-1]
            key_frames.extend([int(i) for i in frame_indices if int(i) not in key_frames])
            key_frames.sort()
        
        # Take the first 5 key frames
        return key_frames[:5]
    
    # Calculate optical flow for motion detection
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitude = np.mean(mag)
            
            # If we have pose data, combine it with motion score
            if has_pose_data and i < len(pose_results):
                pose_data = pose_results[i]
                if pose_data and 'keypoints' in pose_data:
                    # Calculate pose movement score
                    pose_motion = calculate_pose_motion(pose_data)
                    # Combine scores (70% pose, 30% optical flow)
                    motion_magnitude = 0.7 * pose_motion + 0.3 * motion_magnitude
            
            frame_motions.append((i, motion_magnitude))
        prev_frame = gray
    
    if not frame_motions:
        # Fallback to evenly spaced frames
        step = total_frames // 4
        return [0, step, 2*step, 3*step, total_frames-1]
    
    # Enhanced contact detection using combined data
    contact_idx = identify_contact_frame(frames, frame_motions, pose_results if has_pose_data else None)
    
    # Calculate minimum spacing
    min_spacing = max(3, total_frames // 20)
    
    # Enhanced setup frame detection
    setup_idx = identify_setup_frame(frame_motions, contact_idx, pose_results if has_pose_data else None)
    
    # Enhanced load frame detection
    load_idx = identify_load_frame(frame_motions, setup_idx, contact_idx, pose_results if has_pose_data else None)
    
    # Swing frame - just before contact with validation
    swing_idx = identify_swing_frame(frame_motions, load_idx, contact_idx, pose_results if has_pose_data else None)
    
    # Enhanced follow-through detection
    followthrough_idx = identify_followthrough_frame(frame_motions, contact_idx, total_frames, pose_results if has_pose_data else None)
    
    # Ensure indices are properly ordered and within bounds
    indices = [
        min(setup_idx, total_frames - 4),
        min(load_idx, total_frames - 3),
        min(swing_idx, total_frames - 2),
        min(contact_idx, total_frames - 1),
        min(followthrough_idx, total_frames - 1)
    ]
    
    # Validate and adjust frame spacing
    indices = validate_frame_spacing(indices, min_spacing, total_frames)
    
    print(f"Selected frames at: {indices}")
    return indices

def calculate_pose_motion(pose_data):
    """Calculate motion score from pose data"""
    if not pose_data or 'keypoints' not in pose_data:
        return 0.0
        
    # Calculate movement based on key joint positions
    joints = pose_data['keypoints']
    
    # Focus on arms, shoulders, and torso movement
    key_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    
    motion_score = 0.0
    count = 0
    
    for joint in key_joints:
        if joint in joints:
            # Get joint velocity
            if 'velocity' in joints[joint]:
                motion_score += np.linalg.norm(joints[joint]['velocity'])
                count += 1
    
    return motion_score / count if count > 0 else 0.0

def identify_contact_frame(frames, frame_motions, pose_results):
    """Identify contact frame using multiple signals with enhanced accuracy"""
    scores = []
    best_score = 0
    best_idx = None
    
    # First pass: Find frames with significant motion and potential contact
    for i, (idx, motion) in enumerate(frame_motions):
        if idx < len(frames) - 1:  # Ensure we can look ahead
            # Calculate base score from motion
            base_score = motion
            
            # Check bat-ball proximity with higher precision
            is_contact, contact_score = is_contact_frame(frames[idx], frames[idx-1] if idx > 0 else None)
            
            # Enhanced scoring based on multiple factors
            score = base_score
            if is_contact:
                score *= (2.0 + contact_score)  # Increase weight of contact detection
                
                # Validate with pose data if available
                if pose_results and idx < len(pose_results):
                    pose_data = pose_results[idx]
                    if pose_data and 'keypoints' in pose_data:
                        # Check for proper batting form at contact                if is_contact_pose(pose_data):
                            score *= 1.5  # Boost score for correct pose
                            
                            # Additional validation: Check for hip rotation
                            if all(k in pose_data['keypoints'] for k in ['left_hip', 'right_hip']):
                                hip_diff = abs(pose_data['keypoints']['left_hip']['x'] - 
                                            pose_data['keypoints']['right_hip']['x'])
                                if hip_diff > 0.2:  # Significant hip rotation
                                    score *= 1.2
                
                # Look for sudden changes in the next frame
                next_idx = idx + 1
                if next_idx < len(frames):
                    next_ball = detect_baseball(frames[next_idx])
                    if next_ball is not None:
                        curr_ball = detect_baseball(frames[idx])
                        if curr_ball is not None:
                            # Calculate ball movement
                            movement = np.sqrt(
                                (next_ball[0] - curr_ball[0])**2 +
                                (next_ball[1] - curr_ball[1])**2
                            )
                            if movement > 20:  # Significant ball movement after contact
                                score *= 1.3
            
            # Update best score
            if score > best_score:
                best_score = score
                best_idx = idx
                
    # Return best frame or fallback to highest motion
    if best_idx is not None:
        return best_idx
    else:
        # Fallback to frame with maximum motion in the latter half of swing
        latter_half = [(i, m) for i, m in frame_motions if i > len(frame_motions)//2]
        if latter_half:
            return max(latter_half, key=lambda x: x[1])[0]
        return frame_motions[-1][0]  # Last frame as final fallback
        
        scores.append((idx, score))
    
    return max(scores, key=lambda x: x[1])[0]

def identify_setup_frame(frame_motions, contact_idx, pose_results):
    """Identify setup frame using motion and pose data"""
    setup_candidates = [(i, m) for i, m in frame_motions[:contact_idx]]
    
    if pose_results:
        # Filter candidates based on pose
        valid_setups = []
        for idx, motion in setup_candidates:
            if idx < len(pose_results):
                pose_data = pose_results[idx]
                if is_setup_pose(pose_data):
                    valid_setups.append((idx, motion))
        
        if valid_setups:
            return min(valid_setups, key=lambda x: x[1])[0]
    
    return min(setup_candidates, key=lambda x: x[1])[0] if setup_candidates else 0

def identify_load_frame(frame_motions, setup_idx, contact_idx, pose_results):
    """Identify the load frame between setup and contact"""
    # Consider frames between setup and contact
    load_candidates = [(i, m) for i, m in frame_motions if setup_idx < i < contact_idx]
    
    if pose_results:
        # Filter candidates based on pose data
        valid_loads = []
        for idx, motion in load_candidates:
            if idx < len(pose_results) and pose_results[idx]:
                pose_data = pose_results[idx]
                if is_load_pose(pose_data):
                    valid_loads.append((idx, motion))
        
        if valid_loads:
            # Find the frame with maximum motion among valid poses
            return max(valid_loads, key=lambda x: x[1])[0]
    
    # Fallback: take frame at 1/3 distance between setup and contact
    return setup_idx + (contact_idx - setup_idx) // 3

def identify_swing_frame(frame_motions, load_idx, contact_idx, pose_results):
    """Identify the swing frame between load and contact"""
    # Consider frames between load and contact
    swing_candidates = [(i, m) for i, m in frame_motions if load_idx < i < contact_idx]
    
    if pose_results:
        # Filter candidates based on pose data
        valid_swings = []
        for idx, motion in swing_candidates:
            if idx < len(pose_results) and pose_results[idx]:
                pose_data = pose_results[idx]
                if is_swing_pose(pose_data):
                    valid_swings.append((idx, motion))
        
        if valid_swings:
            # Find frame with maximum motion among valid poses
            return max(valid_swings, key=lambda x: x[1])[0]
    
    # Fallback: take frame at 2/3 distance between load and contact
    return load_idx + 2 * (contact_idx - load_idx) // 3

def identify_followthrough_frame(frame_motions, contact_idx, total_frames, pose_results):
    """Identify the follow-through frame after contact"""
    # Consider frames after contact
    followthrough_candidates = [(i, m) for i, m in frame_motions if contact_idx < i < total_frames]
    
    if pose_results:
        # Filter candidates based on pose data
        valid_followthroughs = []
        for idx, motion in followthrough_candidates:
            if idx < len(pose_results) and pose_results[idx]:
                pose_data = pose_results[idx]
                if is_followthrough_pose(pose_data):
                    valid_followthroughs.append((idx, motion))
        
        if valid_followthroughs:
            # Find frame with extended arms and completed rotation
            return max(valid_followthroughs, key=lambda x: x[1])[0]
    
    # Fallback: take frame a few frames after contact
    followthrough_offset = min(15, (total_frames - contact_idx) // 2)
    return min(total_frames - 1, contact_idx + followthrough_offset)

def is_swing_pose(pose_data):
    """Check if pose matches swing position (forward movement and rotation)"""
    if not pose_data or 'keypoints' not in pose_data:
        return False
    
    joints = pose_data['keypoints']
    bat = pose_data.get('bat', {})
    
    # Check for bat movement and body rotation
    if bat.get('grip_position') and 'left_shoulder' in joints and 'right_shoulder' in joints:
        # Check if bat is moving forward and shoulders are rotating
        shoulder_diff = abs(joints['left_shoulder']['x'] - joints['right_shoulder']['x'])
        if shoulder_diff > 0.3 and bat['grip_position']['confidence'] > 0.7:
            return True
    
    return False

def is_followthrough_pose(pose_data):
    """Check if pose matches follow-through position (extended arms and completed rotation)"""
    if not pose_data or 'keypoints' not in pose_data:
        return False
    
    joints = pose_data['keypoints']
    
    # Check for extended arms and completed rotation
    if all(joint in joints for joint in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']):
        # Calculate arm extension
        left_extension = abs(joints['left_wrist']['x'] - joints['left_shoulder']['x'])
        right_extension = abs(joints['right_wrist']['x'] - joints['right_shoulder']['x'])
        
        # Check for full extension and rotation
        if left_extension > 0.3 and right_extension > 0.3:
            return True
    
    return False

def is_contact_pose(pose_data):
    """Check if pose matches contact position with enhanced validation"""
    if not pose_data or 'keypoints' not in pose_data:
        return False
    
    joints = pose_data['keypoints']
    bat = pose_data.get('bat', {})
    
    # Required joints for contact pose validation
    required_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 
                      'right_elbow', 'left_wrist', 'right_wrist',
                      'left_hip', 'right_hip']
    
    # Ensure we have all required joints
    if not all(joint in joints for joint in required_joints):
        return False
      # Enhanced contact pose validation
    try:
        # 1. Check arm extension
        left_arm_extension = (
            abs(joints['left_wrist']['x'] - joints['left_shoulder']['x']) +
            abs(joints['left_wrist']['y'] - joints['left_shoulder']['y'])
        )
        right_arm_extension = (
            abs(joints['right_wrist']['x'] - joints['right_shoulder']['x']) +
            abs(joints['right_wrist']['y'] - joints['right_shoulder']['y'])
        )
        
        # 2. Check shoulder rotation
        shoulder_diff = abs(joints['left_shoulder']['x'] - joints['right_shoulder']['x'])
        
        # 3. Verify bat position if available
        bat_valid = False
        if bat.get('grip_position'):
            bat_pos = bat['grip_position']
            wrists_midpoint_x = (joints['left_wrist']['x'] + joints['right_wrist']['x']) / 2
            wrists_midpoint_y = (joints['left_wrist']['y'] + joints['right_wrist']['y']) / 2
            
            # Bat should be in front of wrists at contact
            bat_valid = (
                bat_pos['x'] > wrists_midpoint_x and
                abs(bat_pos['y'] - wrists_midpoint_y) < 0.2 and  # Bat roughly at same height as wrists
                bat_pos['confidence'] > 0.7
            )
        
        # Combine all factors
        is_valid = (
            left_arm_extension > 0.25 and        # Arms should be extended
            right_arm_extension > 0.25 and
            shoulder_diff > 0.15 and             # Shoulders should be rotated
            (bat_valid if bat.get('grip_position') else True)  # Bat position check if available
        )
        
        return is_valid
    except Exception as e:
        print(f"Error in contact pose validation: {str(e)}")
        return False

def validate_frame_spacing(indices, min_spacing, total_frames):
    """Validate and adjust frame spacing to ensure proper sequence"""
    # First pass: Ensure minimum spacing
    for i in range(1, len(indices)):
        if indices[i] <= indices[i-1] + min_spacing:
            indices[i] = min(total_frames - 1, indices[i-1] + min_spacing)
    
    # Second pass: If frames are too clustered at the end, redistribute
    if indices[-1] >= total_frames - min_spacing and indices[0] < indices[-1] - 4 * min_spacing:
        total_range = indices[-1] - indices[0]
        indices = [
            indices[0],
            indices[0] + total_range // 4,
            indices[0] + (total_range // 4) * 2,
            indices[0] + (total_range // 4) * 3,
            indices[-1]
        ]
    
    return indices

def wait_with_backoff(retry_delay_seconds, max_retries=3):
    """Implement exponential backoff for rate limits"""
    for attempt in range(max_retries):
        delay = retry_delay_seconds * (2 ** attempt)
        print(f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
        time.sleep(delay)
        
def extract_retry_delay(error_message):
    """Extract retry delay from error message"""
    try:
        if "retry_delay" in error_message:
            # Find the retry delay value in the error message
            start = error_message.find("seconds: ") + 9
            end = error_message.find("}", start)
            if start > 8 and end > start:
                return int(error_message[start:end])
    except:
        pass
    return 60  # Default retry delay

def detect_poses_with_gemini(frames):
    """Detect poses in frames using Google's Gemini Vision API with rate limit handling"""
    try:
        # Configure Gemini API        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash')

        pose_results = []
        max_retries = 3
        
        for i, frame in enumerate(frames):
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create the prompt that asks for specific pose analysis
            prompt = """You are a specialized computer vision model for baseball swing analysis. For this baseball swing image, carefully identify and output the EXACT x,y coordinates (normalized between 0.0-1.0) and confidence scores for ALL key body points and the bat.

IMPORTANT: You MUST identify ALL body joints visible in the image, not just the head. Look carefully for shoulders, elbows, wrists, hips, knees, and feet. For any joint you cannot see clearly, make your best estimate of its position based on human anatomy and baseball stance.

Return ONLY a valid JSON object with this exact structure - no other text, no markdown formatting:
{"keypoints":{"head":{"x":0.5,"y":0.2,"confidence":0.9},"left_shoulder":{"x":0.4,"y":0.3,"confidence":0.9},"right_shoulder":{"x":0.6,"y":0.3,"confidence":0.9},"left_elbow":{"x":0.3,"y":0.4,"confidence":0.9},"right_elbow":{"x":0.7,"y":0.4,"confidence":0.9},"left_wrist":{"x":0.2,"y":0.5,"confidence":0.9},"right_wrist":{"x":0.8,"y":0.5,"confidence":0.9},"left_hip":{"x":0.45,"y":0.6,"confidence":0.9},"right_hip":{"x":0.55,"y":0.6,"confidence":0.9},"left_knee":{"x":0.4,"y":0.8,"confidence":0.9},"right_knee":{"x":0.6,"y":0.8,"confidence":0.9},"left_foot":{"x":0.4,"y":0.95,"confidence":0.9},"right_foot":{"x":0.6,"y":0.95,"confidence":0.9}},"bat":{"grip_position":{"x":0.7,"y":0.5,"confidence":0.9},"bat_angle":45,"confidence":0.9}}

Replace these example values with the actual coordinates and confidence scores you detect in the image. For joints that are partially visible or occluded, provide your best estimate and set a lower confidence score (0.5-0.7)."""
            
            for attempt in range(max_retries):
                try:            # Call Gemini API with image content
                    response = model.generate_content([{
                        "text": prompt
                    }, {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }],
                    generation_config={
                        "temperature": 0.1,
                        "candidate_count": 1,
                        "stop_sequences": ["}"],
                        "top_p": 0.8,
                        "top_k": 40
                    },
                    stream=False,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ])# Parse response
                    response_text = response.text
                    # Handle markdown code blocks in the response
                    if "```json" in response_text:
                        # Remove markdown code block formatting
                        response_text = response_text.replace("```json", "").replace("```", "").strip()
                    
                    # More robust JSON extraction
                    try:
                        json_start = response_text.find('{')
                        if json_start >= 0:
                            # Complete the JSON if it appears to be truncated
                            if '}' not in response_text[json_start:]:
                                # Try to complete the JSON structure based on what we have
                                json_content = response_text[json_start:]
                                # Count opening braces to determine how many closing braces to add
                                open_braces = json_content.count('{')
                                close_braces = json_content.count('}')
                                missing_braces = open_braces - close_braces
                                if missing_braces > 0:
                                    json_content += '}' * missing_braces
                            else:
                                # Find the matching closing brace by counting braces
                                brace_count = 0
                                json_end = -1
                                for i in range(json_start, len(response_text)):
                                    if response_text[i] == '{':
                                        brace_count += 1
                                    elif response_text[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i + 1
                                            break
                                
                                if json_end > json_start:
                                    json_content = response_text[json_start:json_end]
                                else:
                                    raise ValueError("Could not find matching closing brace in JSON response")
                            
                            pose_data = json.loads(json_content)
                        else:
                            raise ValueError("Could not find opening brace in JSON response")
                    except Exception as json_error:
                        print(f"Error parsing JSON from response: {str(json_error)}")
                        print(f"Response text: {response_text[:100]}...")  # Print first 100 chars for debugging
                        raise ValueError(f"Failed to parse JSON: {str(json_error)}")
                    
                    # Validate the pose data structure
                    if 'keypoints' in pose_data:
                        required_fields = ['x', 'y', 'confidence']
                        is_valid = all(
                            all(field in joint for field in required_fields)
                            for joint in pose_data['keypoints'].values()
                        )
                        
                        # Check if we have at least the head keypoint
                        if is_valid:
                            # If we only have the head keypoint, add estimated positions for other keypoints
                            if len(pose_data['keypoints']) == 1 and 'head' in pose_data['keypoints']:
                                head = pose_data['keypoints']['head']
                                head_x, head_y = head['x'], head['y']
                                confidence = 0.7  # Lower confidence for estimated keypoints
                                
                                # Add estimated positions for other keypoints based on head position
                                # These are rough anatomical estimates for a baseball batter
                                pose_data['keypoints']['left_shoulder'] = {'x': head_x - 0.05, 'y': head_y + 0.1, 'confidence': confidence}
                                pose_data['keypoints']['right_shoulder'] = {'x': head_x + 0.05, 'y': head_y + 0.1, 'confidence': confidence}
                                pose_data['keypoints']['left_elbow'] = {'x': head_x - 0.1, 'y': head_y + 0.2, 'confidence': confidence}
                                pose_data['keypoints']['right_elbow'] = {'x': head_x + 0.15, 'y': head_y + 0.2, 'confidence': confidence}
                                pose_data['keypoints']['left_wrist'] = {'x': head_x - 0.15, 'y': head_y + 0.3, 'confidence': confidence}
                                pose_data['keypoints']['right_wrist'] = {'x': head_x + 0.2, 'y': head_y + 0.3, 'confidence': confidence}
                                pose_data['keypoints']['left_hip'] = {'x': head_x - 0.05, 'y': head_y + 0.4, 'confidence': confidence}
                                pose_data['keypoints']['right_hip'] = {'x': head_x + 0.05, 'y': head_y + 0.4, 'confidence': confidence}
                                pose_data['keypoints']['left_knee'] = {'x': head_x - 0.05, 'y': head_y + 0.6, 'confidence': confidence}
                                pose_data['keypoints']['right_knee'] = {'x': head_x + 0.05, 'y': head_y + 0.6, 'confidence': confidence}
                                pose_data['keypoints']['left_foot'] = {'x': head_x - 0.05, 'y': head_y + 0.8, 'confidence': confidence}
                                pose_data['keypoints']['right_foot'] = {'x': head_x + 0.05, 'y': head_y + 0.8, 'confidence': confidence}
                                
                                # Add estimated bat position
                                if 'bat' not in pose_data:
                                    pose_data['bat'] = {
                                        'grip_position': {'x': head_x + 0.2, 'y': head_y + 0.3, 'confidence': confidence},
                                        'bat_angle': 45,
                                        'confidence': confidence
                                    }
                            
                            print(f"Frame {i}: Successfully detected {len(pose_data['keypoints'])} keypoints")
                            pose_results.append(pose_data)
                        else:
                            print(f"Frame {i}: Invalid keypoint data structure")
                            pose_results.append(None)
                    else:
                        print(f"Frame {i}: Missing keypoints in response")
                        pose_results.append(None)
                    # Successfully processed frame, break retry loop
                    break
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message:  # Rate limit error
                        retry_delay = extract_retry_delay(error_message)
                        if attempt < max_retries - 1:  # Don't wait on last attempt
                            wait_with_backoff(retry_delay, max_retries - attempt)
                            continue
                    print(f"Error processing frame {i}: {error_message}")
                    pose_results.append(None)
                    break

            # Add a small delay between frames to avoid rate limits
            time.sleep(1)

        return pose_results

    except Exception as e:
        print(f"Error in detect_poses_with_gemini: {str(e)}")
        return None

def extract_frames(video_path, num_frames=5, max_memory_mb=500):
    """Extract key frames from a video file.
    
    Memory Optimization:
    - Streaming frame processing instead of loading all frames
    - Frame buffering for efficient processing
    - Batch processing for large videos
    - Memory usage monitoring
    - Frame quality validation
    """
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open video file with OpenCV")
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {frame_count} frames, {fps} fps, {width}x{height}")
    
    # Calculate memory requirements
    frame_size_bytes = width * height * 3  # 3 bytes per pixel (BGR)
    frame_size_mb = frame_size_bytes / (1024 * 1024)
    
    # Determine if we need batch processing
    use_batch_processing = (frame_size_mb * frame_count) > max_memory_mb
    
    if use_batch_processing:
        print(f"Using batch processing due to memory constraints. Frame size: {frame_size_mb:.2f} MB")
        batch_size = max(10, int(max_memory_mb / frame_size_mb))
        print(f"Batch size: {batch_size} frames")
        key_frames = extract_frames_batch(cap, frame_count, batch_size, num_frames)
    else:
        print("Processing all frames in memory")
        key_frames = extract_frames_full(cap, frame_count, num_frames)
    
    cap.release()
    return key_frames

def extract_frames_batch(cap, frame_count, batch_size, num_frames=5):
    """Process video in batches to reduce memory usage"""
    # Initialize variables for motion tracking
    motion_scores = []
    frame_buffer = []
    batch_start = 0
    
    # Process video in batches
    while batch_start < frame_count:
        print(f"Processing batch starting at frame {batch_start}")
        
        # Set position to batch start
        cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
        
        # Process batch
        batch_frames = []
        batch_end = min(batch_start + batch_size, frame_count)
        prev_frame = None
        
        for _ in range(batch_start, batch_end):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (640, 360))
            
            # Validate frame quality
            quality = validate_frame_quality(frame)
            
            if quality > 0.5:  # Only keep frames with good quality
                batch_frames.append(frame)
                
                # Calculate motion score if we have a previous frame
                if prev_frame is not None:
                    motion = calculate_frame_motion(prev_frame, frame)
                    motion_scores.append((len(motion_scores), motion))
                
                prev_frame = frame.copy()
        
        # Update frame buffer with best frames from this batch
        update_frame_buffer(frame_buffer, batch_frames, motion_scores, batch_start)
        
        # Move to next batch
        batch_start = batch_end
    
    # Select key frames from the buffer
    if len(frame_buffer) < num_frames:
        print(f"Warning: Only found {len(frame_buffer)} good quality frames")
        # Pad with duplicates if needed
        while len(frame_buffer) < num_frames:
            if frame_buffer:
                frame_buffer.append(frame_buffer[-1])
            else:
                break
        return frame_buffer
    
    # Use motion analysis to find key frames
    return select_key_frames(frame_buffer, motion_scores, num_frames)

def extract_frames_full(cap, frame_count, num_frames=5):
    """Process all video frames in memory"""
    # Read all frames
    all_frames = []
    motion_scores = []
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (640, 360))
        
        # Validate frame quality
        quality = validate_frame_quality(frame)
        
        if quality > 0.5:  # Only keep frames with good quality
            all_frames.append(frame)
            
            # Calculate motion score if we have a previous frame
            if prev_frame is not None:
                motion = calculate_frame_motion(prev_frame, frame)
                motion_scores.append((len(motion_scores), motion))
            
            prev_frame = frame.copy()
    
    if len(all_frames) < num_frames:
        print(f"Warning: Only found {len(all_frames)} good quality frames")
        # Use regular interval sampling for very short videos
        step = max(1, len(all_frames) // num_frames)
        indices = [i * step for i in range(num_frames) if i * step < len(all_frames)]
        while len(indices) < num_frames:
            indices.append(len(all_frames) - 1)
        return [all_frames[i] for i in indices[:num_frames]]
    
    # Use motion analysis to find key frames
    return select_key_frames(all_frames, motion_scores, num_frames)

def validate_frame_quality(frame):
    """Validate frame quality using blur detection and exposure checks"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check for blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 500)  # Normalize, higher is better
    
    # Check for proper exposure
    mean_brightness = np.mean(gray)
    exposure_score = 1.0 - abs(mean_brightness - 128) / 128  # 128 is middle gray
    
    # Check for contrast
    std_brightness = np.std(gray)
    contrast_score = min(1.0, std_brightness / 50)  # Normalize, higher is better
    
    # Combined quality score
    quality_score = 0.5 * blur_score + 0.3 * exposure_score + 0.2 * contrast_score
    
    return quality_score

def calculate_frame_motion(prev_frame, curr_frame):
    """Calculate motion between two frames"""
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Calculate motion score
    motion = np.mean(diff)
    
    return motion

def update_frame_buffer(frame_buffer, batch_frames, motion_scores, batch_start):
    """Update frame buffer with best frames from the current batch"""
    # Keep frames with highest motion scores
    if batch_frames:
        # Get motion scores for this batch
        batch_motion = [score for idx, score in motion_scores if idx >= batch_start]
        
        # If we have motion scores for this batch
        if batch_motion:
            # Find frames with highest motion
            threshold = np.mean(batch_motion) + 0.5 * np.std(batch_motion)
            high_motion_indices = [i for i, motion in enumerate(batch_motion) if motion > threshold]
            
            # Add high motion frames to buffer
            for idx in high_motion_indices:
                if idx < len(batch_frames):
                    frame_buffer.append(batch_frames[idx])
        
        # Always keep first and last frame of batch
        if batch_frames and len(batch_frames) > 0:
            if not any(np.array_equal(batch_frames[0], f) for f in frame_buffer):
                frame_buffer.append(batch_frames[0])
                
        if batch_frames and len(batch_frames) > 1:
            if not any(np.array_equal(batch_frames[-1], f) for f in frame_buffer):
                frame_buffer.append(batch_frames[-1])

def select_key_frames(frames, motion_scores, num_frames=5):
    """Select key frames using motion analysis"""
    if len(frames) < 5:
        return frames
    
    print("Analyzing swing mechanics for key frame selection...")
    
    # Use motion detection to find key swing phases
    indices = detect_swing_phase(frames)
    
    # Ensure we have exactly the requested number of frames
    if len(indices) > num_frames:
        indices = indices[:num_frames]
    while len(indices) < num_frames:
        # Find largest gap and add frame in middle
        max_gap = 0
        insert_idx = 0
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i
        indices.insert(insert_idx + 1, indices[insert_idx] + (indices[insert_idx + 1] - indices[insert_idx]) // 2)
    
    # Sort indices to maintain temporal order
    indices.sort()
    key_frames = [frames[i] for i in indices]
    
    print(f"Selected frame indices: {indices}")
    
    return key_frames




def save_frames_to_s3(frames, analysis_id):
    """Save extracted frames to S3.
    
    Storage Optimization:
    - Frame compression with quality control
    - Progressive loading support
    - Caching layer implementation
    - Optimized metadata storage
    """
    frame_paths = []
    frame_urls = []
    metadata = {}

    print(f"Saving {len(frames)} frames to S3 for analysis {analysis_id}")
    
    # Create metadata structure for optimized storage
    metadata = {
        "analysis_id": analysis_id,
        "frame_count": len(frames),
        "created_at": int(time.time()),
        "frames": []
    }
    
    # Define compression quality levels for different sizes
    # Higher quality for thumbnail, lower for full size to save bandwidth
    quality_levels = {
        "thumbnail": 80,  # Higher quality for small images
        "medium": 75,     # Medium quality for preview
        "full": 70        # Lower quality for full size to save storage
    }
    
    # Define sizes for progressive loading
    sizes = {
        "thumbnail": (160, 90),   # 16:9 aspect ratio, small thumbnail
        "medium": (320, 180),     # Medium preview
        "full": (640, 360)        # Full analysis size
    }
    
    for i, frame in enumerate(frames):
        try:
            frame_variants = {}
            frame_variant_urls = {}
            
            # Generate different sizes for progressive loading
            for size_name, dimensions in sizes.items():
                # Resize the frame
                if size_name != "full":
                    resized = cv2.resize(frame, dimensions)
                else:
                    resized = frame
                
                # Compress with appropriate quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_levels[size_name]]
                _, buffer = cv2.imencode('.jpg', resized, encode_param)
                
                # Upload to S3
                variant_key = f"analyses/{analysis_id}/frames/frame_{i}_{size_name}.jpg"
                
                # Add cache control headers for browser caching
                cache_control = "max-age=31536000" if size_name != "full" else "max-age=3600"
                
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=variant_key,
                    Body=buffer.tobytes(),
                    ContentType='image/jpeg',
                    CacheControl=cache_control
                )
                
                # Generate presigned URL with appropriate expiration
                expiration = 86400 if size_name == "thumbnail" else 3600  # 24 hours for thumbnails, 1 hour for others
                url = get_presigned_url(variant_key, expiration)
                
                # Store paths and URLs
                frame_variants[size_name] = variant_key
                if url:
                    frame_variant_urls[size_name] = url
            
            # Add the main frame path to the list (full size version)
            frame_paths.append(frame_variants["full"])
            if "full" in frame_variant_urls:
                frame_urls.append(frame_variant_urls["full"])
            
            # Add frame metadata
            frame_metadata = {
                "index": i,
                "variants": frame_variants,
                "urls": frame_variant_urls,
                "size_bytes": {
                    size: len(cv2.imencode('.jpg', 
                                          cv2.resize(frame, dimensions) if size != "full" else frame,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), quality_levels[size]])[1])
                    for size, dimensions in sizes.items()
                }
            }
            
            metadata["frames"].append(frame_metadata)
            print(f"Saved frame {i} with progressive loading variants")
            
        except Exception as e:
            print(f"Error saving frame {i}: {str(e)}")
            raise
    
    # Save consolidated metadata
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/frames_metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        print(f"Saved consolidated frame metadata for {analysis_id}")
    except Exception as e:
        print(f"Error saving frame metadata: {str(e)}")
    
    print(f"Successfully saved {len(frame_paths)} frames to S3")
    return frame_paths, frame_urls

def get_reference_frames(player_id, num_frames=5):
    """Get reference frames for the specified player"""
    reference_frames = []
    reference_urls = []

    try:
        # First try to get frames from metadata
        try:
            metadata_key = f"reference/{player_id}/metadata.json"
            response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            
            if 'frame_paths' in metadata:
                print(f"Found metadata with {len(metadata['frame_paths'])} frame paths for player {player_id}")
                frame_keys = metadata['frame_paths']
                
                # If we have frame_urls in metadata, use them
                if 'frame_urls' in metadata and len(metadata['frame_urls']) == len(frame_keys):
                    reference_urls = metadata['frame_urls']
            else:
                print(f"Metadata found but no frame_paths for player {player_id}")
                frame_keys = []
        except Exception as e:
            print(f"Error loading metadata, falling back to listing objects: {str(e)}")
            # List reference frames for the player
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f"reference/{player_id}/frames/"
            )
            
            if 'Contents' in response:
                # Sort by key to ensure frames are in order
                frame_keys = sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.jpg') and 'full' in obj['Key']])
                
                if not frame_keys:
                    # Fall back to old format if no 'full' variant frames found
                    frame_keys = sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.jpg')])
            else:
                print(f"No reference frames found for player {player_id}")
                frame_keys = []
        
        # Select evenly spaced frames if there are more than we need
        if len(frame_keys) > num_frames:
            step = len(frame_keys) // num_frames
            frame_keys = [frame_keys[i] for i in range(0, len(frame_keys), step)][:num_frames]
        
        print(f"Found {len(frame_keys)} reference frames for player {player_id}")
        
        # Download each frame
        for key in frame_keys:
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                image_data = response['Body'].read()
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Resize for consistency
                    img = cv2.resize(img, (640, 360))
                    reference_frames.append(img)
                    
                    # Generate URL if we don't already have it
                    if len(reference_urls) < len(reference_frames):
                        url = get_presigned_url(key)
                        if url:
                            reference_urls.append(url)
                    
                    print(f"Loaded reference frame: {key}")
                else:
                    print(f"Failed to decode reference frame: {key}")
            except Exception as frame_e:
                print(f"Error loading frame {key}: {str(frame_e)}")
                continue
    except Exception as e:
        print(f"Error loading reference frames: {str(e)}")
        print(traceback.format_exc())
    
    return reference_frames, reference_urls

def get_presigned_url(key, expiration=3600):
    """Generate a presigned URL for an S3 object"""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': key
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL for {key}: {str(e)}")
        return None


def compare_frames(user_frames, reference_frames):
    """Compare user frames with reference frames using advanced image processing.
    
    TODO: Enhance Comparison
    - Re-enable pose difference calculation
    - Add joint angle analysis
    - Implement quantitative metrics
    - Generate specific improvement suggestions
    - Add visual alignment overlays
    """
    if not reference_frames:
        print("No reference frames available for comparison")
        return None
    
    # Ensure we have the same number of frames to compare
    min_frames = min(len(user_frames), len(reference_frames))
    user_frames = user_frames[:min_frames]
    reference_frames = reference_frames[:min_frames]    
    comparison_results = []
    for i, (user_frame, ref_frame) in enumerate(zip(user_frames, reference_frames)):
        # Create ROI masks based on center region where batter is likely to be
        h, w = user_frame.shape[:2]
        center_x = w // 2
        roi_width = w // 2  # Use middle 50% of frame width
        user_mask = np.zeros_like(cv2.cvtColor(user_frame, cv2.COLOR_BGR2GRAY))
        ref_mask = np.zeros_like(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY))
        user_mask[:, center_x - roi_width//2:center_x + roi_width//2] = 255
        ref_mask[:, center_x - roi_width//2:center_x + roi_width//2] = 255
          # Create copies for annotation
        user_annotated = user_frame.copy()
        ref_annotated = ref_frame.copy()
        
        # Normalize lighting and process frames
        user_lab = cv2.cvtColor(user_frame, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2LAB)
        
        # Normalize L channel (lighting)
        user_l = cv2.normalize(user_lab[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        ref_l = cv2.normalize(ref_lab[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        
        # Analyze bat position
        user_bat = detect_bat_region(user_frame)
        ref_bat = detect_bat_region(ref_frame)
        
        # Draw bat position annotations
        if user_bat:
            cv2.rectangle(user_annotated, 
                         (user_bat['x'], user_bat['y']), 
                         (user_bat['x'] + user_bat['w'], user_bat['y'] + user_bat['h']), 
                         (0, 255, 0), 2)
            
        if ref_bat:
            cv2.rectangle(ref_annotated,
                         (ref_bat['x'], ref_bat['y']),
                         (ref_bat['x'] + ref_bat['w'], ref_bat['y'] + ref_bat['h']),
                         (0, 255, 0), 2)        # Pose difference visualization temporarily disabled
        pose_difference = np.zeros_like(user_l)
        significant_diff = np.zeros_like(pose_difference)
        
        # Skip overlay for now since we're not using advanced isolation
        user_annotated = user_frame.copy()
        ref_annotated = ref_frame.copy()        # Calculate edge ratio for stance analysis using masked frames
        user_gray = cv2.cvtColor(user_frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection with auto-threshold
        user_median = np.median(user_gray)
        ref_median = np.median(ref_gray)
        user_sigma = 0.33
        ref_sigma = 0.33
        
        # Compute thresholds based on median pixel values
        user_lower = int(max(0, (1.0 - user_sigma) * user_median))
        user_upper = int(min(255, (1.0 + user_sigma) * user_median))
        ref_lower = int(max(0, (1.0 - ref_sigma) * ref_median))
        ref_upper = int(min(255, (1.0 + ref_sigma) * ref_median))
        
        # Apply edge detection with dynamic thresholds
        user_edges = cv2.Canny(user_gray, user_lower, user_upper)
        ref_edges = cv2.Canny(ref_gray, ref_lower, ref_upper)
          # Use whole frame edge detection temporarily instead of isolated regions
        user_edge_count = np.count_nonzero(user_edges)
        ref_edge_count = np.count_nonzero(ref_edges)
        edge_ratio = min(user_edge_count, ref_edge_count) / max(user_edge_count, ref_edge_count) if max(user_edge_count, ref_edge_count) > 0 else 0
          # Initialize lists for annotations and issues
        annotations = []
        issues = []
        
        # Analyze stance width and height
        user_stance = detect_batter_position(user_frame)
        ref_stance = detect_batter_position(ref_frame)
        
        if user_stance and ref_stance:
            # Compare stance dimensions
            user_width = user_stance[2]  # width
            user_height = user_stance[3]  # height
            ref_width = ref_stance[2]
            ref_height = ref_stance[3]
            
            # Calculate ratios relative to frame size
            user_width_ratio = user_width / user_frame.shape[1]
            user_height_ratio = user_height / user_frame.shape[0]
            ref_width_ratio = ref_width / ref_frame.shape[1]
            ref_height_ratio = ref_height / ref_frame.shape[0]
            
            # Compare stance dimensions
            width_diff = (user_width_ratio - ref_width_ratio) / ref_width_ratio
            height_diff = (user_height_ratio - ref_height_ratio) / ref_height_ratio
            
            # Add stance-specific annotations
            if abs(width_diff) > 0.15:  # More than 15% difference
                desc = "too wide" if width_diff > 0 else "too narrow"
                annotations.append({
                    'type': 'stance_width',
                    'magnitude': abs(width_diff),
                    'description': f'Batting stance is {desc}',
                    'details': f'Your stance is {abs(width_diff)*100:.0f}% {desc} compared to the reference'
                })
            
            if abs(height_diff) > 0.15:  # More than 15% difference
                desc = "too upright" if height_diff < 0 else "too crouched"
                annotations.append({
                    'type': 'stance_height',
                    'magnitude': abs(height_diff),
                    'description': f'Batting stance is {desc}',
                    'details': f'Your stance is {abs(height_diff)*100:.0f}% {desc} compared to the reference'
                })
        
        # Calculate basic stance similarity
        stance_diff = abs(user_edge_count - ref_edge_count) / max(user_edge_count, ref_edge_count)
        stance_similarity = 1.0 - stance_diff  # Keep as decimal between 0 and 1        # Calculate histogram similarity using center ROI only
        user_hist = cv2.calcHist([user_frame], [0, 1, 2], user_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        ref_hist = cv2.calcHist([ref_frame], [0, 1, 2], ref_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(user_hist, user_hist)
        cv2.normalize(ref_hist, ref_hist)
        hist_similarity = max(0, min(1, cv2.compareHist(user_hist, ref_hist, cv2.HISTCMP_CORREL)))
        # Apply a stricter threshold to histogram similarity
        hist_similarity = hist_similarity * 0.8  # Reduce weight of histogram similarity
        
        # Calculate pose similarity from difference matrix
        pose_similarity = 1.0 - (np.sum(pose_difference) / (pose_difference.shape[0] * pose_difference.shape[1] * 255))
          # Combined weighted score with phase-specific weights
        if i == 0:  # Setup phase - emphasize stance and posture
            similarity_score = 0.3 * pose_similarity + 0.2 * hist_similarity + 0.5 * stance_similarity
        elif i == len(user_frames) - 1:  # Follow-through - emphasize pose and form
            similarity_score = 0.5 * pose_similarity + 0.3 * hist_similarity + 0.2 * stance_similarity
        else:  # Other phases - balanced scoring
            similarity_score = 0.4 * pose_similarity + 0.3 * hist_similarity + 0.3 * stance_similarity
            
        # Ensure score is between 0 and 1
        similarity_score = max(0, min(1, similarity_score))
        
        # Generate annotations and issues based on specific thresholds
        annotations = []
        issues = []
        
        # Add pose difference annotations with specific thresholds
        if pose_similarity < 0.7:
            magnitude = 1.0 - pose_similarity
            desc = 'Minor' if magnitude < 0.4 else 'Significant' if magnitude < 0.6 else 'Major'
            annotations.append({
                'type': 'pose_difference',
                'magnitude': magnitude,
                'description': f'{desc} pose difference detected'
            })
            issues.append({
                'type': 'pose',
                'severity': desc.lower(),
                'description': f'{desc} body position adjustment needed to match the reference'
            })
          # Add bat position annotations with detailed analysis
        if user_bat and ref_bat:
            # Calculate differences in position and angle
            bat_x_diff = abs(user_bat['x'] - ref_bat['x'])
            bat_y_diff = abs(user_bat['y'] - ref_bat['y'])
            
            # Normalize differences relative to frame size
            norm_x_diff = bat_x_diff / user_frame.shape[1]
            norm_y_diff = bat_y_diff / user_frame.shape[0]
            
            # Calculate weighted bat position difference
            bat_diff = np.sqrt(norm_x_diff**2 + norm_y_diff**2)
            
            # Determine severity of bat position difference
            if bat_diff > 0.05:  # More than 5% of frame size
                severity = 'Minor' if bat_diff < 0.1 else 'Significant' if bat_diff < 0.2 else 'Major'
                description = f'{severity} bat position difference detected'
                
                # Add detailed position analysis
                if norm_x_diff > norm_y_diff:
                    description += ' - horizontal alignment needs adjustment'
                else:
                    description += ' - vertical alignment needs adjustment'
                
                annotations.append({
                    'type': 'bat_position',
                    'magnitude': float(bat_diff),
                    'region': {'x': user_bat['x'], 'y': user_bat['y'], 'w': user_bat['w'], 'h': user_bat['h']},
                    'description': description
                })
        
        # Generate drills based on identified issues
        drills = []
        if similarity_score < 0.6:
            drills.append({
                'type': 'form',
                'name': 'Mirror Practice',
                'description': 'Practice your swing in front of a mirror, focusing on matching the reference pose',
                'steps': [
                    'Set up in front of a full-length mirror',
                    'Take your stance and compare to the reference image',
                    'Practice the swing in slow motion, checking your form at each phase',
                    'Focus on matching the key positions shown in the reference'
                ]
            })
        
        if any(a['type'] == 'bat_position' for a in annotations):
            drills.append({
                'type': 'bat_control',
                'name': 'Bat Path Drill',
                'description': 'Practice maintaining proper bat path and position through the swing',
                'steps': [
                    'Set up with a tee at proper contact height',
                    'Place visual markers at key positions (setup, load, contact)',
                    'Focus on keeping the bat on the proper path through these positions',
                    'Practice with slow, controlled movements initially'
                ]
            })
          # Calculate detailed metrics
        detailed_metrics = {
            'pose_similarity': float(pose_similarity),
            'stance_similarity': float(stance_similarity),
            'hist_similarity': float(hist_similarity),
            'edge_ratio': float(edge_ratio)
        }
        
        # Determine swing phase
        phase_names = ['Setup', 'Load', 'Swing', 'Contact', 'Follow-through']
        phase_name = phase_names[i] if i < len(phase_names) else f'Frame {i}'
        
        comparison_results.append({
            'frame_index': i,
            'phase_name': phase_name,
            'similarity_score': float(similarity_score),
            'detailed_metrics': detailed_metrics,
            'annotations': annotations,
            'issues': issues,
            'drills': drills,
            'user_annotated': base64.b64encode(cv2.imencode('.jpg', user_annotated)[1].tobytes()).decode('utf-8'),
            'ref_annotated': base64.b64encode(cv2.imencode('.jpg', ref_annotated)[1].tobytes()).decode('utf-8')
        })
    
    return comparison_results

def analyze_swing(user_frames, reference_frames, comparison_results):
    """Generate comprehensive swing analysis.
    
    TODO: Feedback Enhancement
    - Add quantitative measurements
    - Include specific angle comparisons
    - Generate frame-specific drills
    - Add visual feedback markers
    - Implement progression tracking
    """
    if not comparison_results:
        return {
            "overall_score": 50,
            "strengths": ["Consistent swing pattern"],
            "areas_to_improve": ["Work on matching professional form"],
            "detailed_feedback": "We couldn't perform a detailed comparison with the reference player. Try uploading a clearer video.",
            "drills": [{
                "name": "Basic Mirror Work",
                "description": "Practice your swing in front of a mirror to develop muscle memory",
                "steps": [
                    "Set up in front of a full-length mirror",
                    "Take your stance and check your alignment",
                    "Practice your swing in slow motion",
                    "Focus on fluid movement and balance"
                ]
            }]
        }
    
    # Calculate overall score
    overall_score = int(sum(result["similarity_score"] for result in comparison_results) / len(comparison_results) * 100)
    
    # Analyze annotations across all frames
    all_annotations = [ann for result in comparison_results for ann in result["annotations"]]
    
    # Group issues by type
    issue_types = {}
    for annotation in all_annotations:
        if annotation["type"] not in issue_types:
            issue_types[annotation["type"]] = []
        issue_types[annotation["type"]].append(annotation)
    
    # Collect all recommended drills
    all_drills = [drill for result in comparison_results for drill in result["drills"]]
    unique_drills = []
    drill_names = set()
    
    for drill in all_drills:
        if drill["name"] not in drill_names:
            drill_names.add(drill["name"])
            unique_drills.append({
                "name": drill["name"],
                "description": drill["description"],
                "steps": drill["steps"]
            })
    
    # Determine strengths and areas to improve with specific feedback
    strengths = []
    areas_to_improve = []
    
    # Add strengths based on high similarity scores and good form
    high_scores = [r for r in comparison_results if r["similarity_score"] > 0.7]
    if len(high_scores) > len(comparison_results) / 2:
        strengths.append({
            "text": "Good overall form matching the professional reference",
            "details": "Your stance and movement pattern closely matches the reference"
        })
    
    if len(high_scores) > 0:
        best_frame = max(comparison_results, key=lambda x: x["similarity_score"])
        if best_frame["frame_index"] == 0:
            strengths.append({
                "text": "Excellent setup position",
                "details": "Your initial stance demonstrates proper balance and readiness"
            })
        elif best_frame["frame_index"] == len(comparison_results) - 1:
            strengths.append({
                "text": "Strong follow-through",
                "details": "You complete your swing with good extension and balance"
            })
    
    # Analyze pose differences
    pose_issues = issue_types.get("pose_difference", [])
    if pose_issues:
        avg_difference = sum(issue["magnitude"] for issue in pose_issues) / len(pose_issues)
        if avg_difference > 0.3:
            areas_to_improve.append({
                "text": "Work on your body positioning throughout the swing",
                "details": "Focus on maintaining proper posture and alignment",
                "drill": {
                    "name": "Posture Alignment Drill",
                    "description": "Improve your swing posture and body alignment",
                    "steps": [
                        "Set up in front of a mirror",
                        "Place alignment rods on the ground",
                        "Practice maintaining proper spine angle",
                        "Check alignment at key positions: setup, load, contact"
                    ]
                }
            })
    
    # Analyze bat path
    bat_position_issues = issue_types.get("bat_position", [])
    if bat_position_issues:
        avg_magnitude = sum(issue["magnitude"] for issue in bat_position_issues) / len(bat_position_issues)
        if avg_magnitude > 0.2:
            areas_to_improve.append({
                "text": "Improve your bat path consistency",
                "details": "Work on maintaining a level swing plane",
                "drill": {
                    "name": "Level Swing Path Drill",
                    "description": "Develop a more consistent and level bat path",
                    "steps": [
                        "Set up with a tee at belt height",
                        "Place visual markers at contact point",
                        "Focus on keeping the bat level through the zone",
                        "Practice with slow, controlled movements"
                    ]
                }
            })
    
    # Add timing drill if needed
    if overall_score < 70:
        unique_drills.append({
            "name": "Timing Refinement Drill",
            "description": "Improve your swing timing and rhythm",
            "steps": [
                "Use a tee or soft toss",
                "Practice with a metronome",
                "Focus on smooth load and trigger movements",
                "Gradually increase speed while maintaining form"
            ]
        })
    
    # Ensure we have at least one strength
    if not strengths:
        if overall_score > 60:
            strengths.append({
                "text": "Consistent swing mechanics",
                "details": "You maintain good rhythm through your swing"
            })
        else:
            strengths.append({
                "text": "Good effort and swing attempt",
                "details": "You're showing commitment to improving your technique"
            })
    
    # Generate detailed feedback
    score_text = "excellent" if overall_score > 80 else "good" if overall_score > 60 else "developing"
    detailed_feedback = f"Your swing shows {score_text} potential with a similarity score of {overall_score}/100. "
    
    if strengths:
        detailed_feedback += f"\n\nKey Strengths:\n"
        for strength in strengths:
            detailed_feedback += f"- {strength['text']}: {strength['details']}\n"
    
    if areas_to_improve:
        detailed_feedback += f"\n\nAreas to Focus On:\n"
        for area in areas_to_improve:
            detailed_feedback += f"- {area['text']}: {area['details']}\n"
    
    # Add drill recommendations
    if unique_drills:
        detailed_feedback += "\n\nRecommended Drills:\n"
        for drill in unique_drills:
            detailed_feedback += f"- {drill['name']}: {drill['description']}\n"
            detailed_feedback += "  Steps:\n"
            for step in drill['steps']:
                detailed_feedback += f"    * {step}\n"
    
    return {
        "overall_score": overall_score,
        "strengths": [s["text"] for s in strengths],
        "areas_to_improve": [a["text"] for a in areas_to_improve],
        "detailed_feedback": detailed_feedback,
        "drills": unique_drills,
        "comparison_results": comparison_results
    }

# TODO: Enhancement Areas - 2025 Roadmap

"""
Major areas for improvement:

1. Batter Isolation
   - Currently disabled (returns original frame)
   - Need to implement deep learning-based segmentation (e.g., Mask R-CNN or DeepLab)
   - Consider creating a custom dataset of baseball players for training
   - Add background subtraction as fallback

2. Parameter Optimization
   - Replace hard-coded thresholds with dynamic/adaptive values:
     detect_baseball(): minRadius/maxRadius
     detect_bat(): aspect ratio thresholds
     compare_frames(): similarity thresholds
   - Implement auto-calibration based on video properties
   - Add configuration file for environment-specific tuning

3. Motion Analysis Enhancements
   - Current: Basic Shi-Tomasi + Lucas-Kanade
   - Upgrade to dense optical flow for better tracking
   - Add deep learning-based pose estimation
   - Implement motion prediction for fast movements

4. Pose Analysis Pipeline
   - Re-enable pose difference calculation
   - Integrate MediaPipe or OpenPose
   - Add specific joint angle measurements
   - Implement quantitative pose metrics

5. Frame Selection And Phase Detection
   - Improve detect_swing_phase() robustness
   - Add ML-based key frame identification
   - Implement temporal clustering
   - Add motion peak analysis

6. Memory Optimization
   - Current: Loads all frames into memory
   - Switch to streaming frame processing
   - Implement batch processing
   - Add memory usage monitoring

7. Feedback Enhancement
   - Add quantitative measurements
   - Include specific angle comparisons
   - Generate frame-specific drills
   - Add video overlay annotations
"""

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
                if 0.8 < aspect_ratio < 1.2 and area < 500: # Ball-like object
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
                # Calculate motion vectors (dx, dy for each point)
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
                    'magnitude': float(avg_magnitude),  # Convert to native Python float
                    'direction': float(avg_direction),  # Convert to native Python float
                    'points': good_new
                })
            
            # Update points
            p0 = good_new.reshape(-1, 1, 2)
    
    return trajectories

def lambda_handler(event, context):
    """Process video to extract frames for analysis"""
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With,Accept,Origin',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
    }
    
    analysis_id = None
    video_key = None
    
    try:
        # Log the event
        print(f"Received event: {json.dumps(event)}")
        
        # Parse the event body if it's a string
        if 'body' in event and isinstance(event['body'], str):
            body = json.loads(event['body'])
        else:
            body = event
        
        # Get video info from event
        analysis_id = body.get('analysis_id')
        player_id = body.get('player_id', 'bryce_harper')
        
        # Get player name from mapping or format player_id as a name
        player_name = PLAYER_NAMES.get(player_id, player_id.replace('_', ' ').title())
        
        if not analysis_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': f"Missing required parameter: analysis_id"
                })
            }
        
        # Get the video key from the request
        video_key = body.get('video_key')
        if not video_key:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': f"Missing required parameter: video_key"
                })
            }
        
        print(f"Using video key: {video_key} for analysis: {analysis_id}")
        
        # Create initial metadata to indicate processing
        metadata = {
            "analysis_id": analysis_id,
            "video_key": video_key,
            "status": "processing",
            "player_id": player_id
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
               
        # Download video from S3
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_path = temp_video.name
            
            try:
                s3_client.download_file(bucket_name, video_key, temp_path)
                print(f"Successfully downloaded video to {temp_path}")
                
                # Extract frames
                frames = extract_frames(temp_path)
            finally:
                # Clean up the temp file
                try:
                    os.unlink(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error cleaning up temp file: {str(e)}")
        
        # Save frames to S3
        frame_paths, frame_urls = save_frames_to_s3(frames, analysis_id)
        
        # Get reference frames for the selected player
        reference_frames, reference_urls = get_reference_frames(player_id)
        
        # Compare user frames with reference frames
        comparison_results = compare_frames(frames, reference_frames)
        
        # Generate comprehensive analysis
        if comparison_results:
            analysis_results = analyze_swing(frames, reference_frames, comparison_results)
            
            # Create feedback based on the analysis
            feedback = {
                "status": "feedback_generated",
                "player_id": player_id,
                "player_name": player_name,  # Use the mapped player name
                "overall_score": analysis_results["overall_score"],
                "strengths": analysis_results["strengths"],
                "areas_to_improve": analysis_results["areas_to_improve"],
                "detailed_feedback": analysis_results["detailed_feedback"],
                "comparison_results": analysis_results["comparison_results"]
            }
        else:
            # Fallback to basic analysis if no reference frames or comparison failed
            print("No comparison results available, using basic analysis")
            feedback = {
                "status": "feedback_generated",
                "player_id": player_id,
                "player_name": player_name,  # Use the mapped player name
                "overall_score": 65,
                "strengths": ["Consistent swing pattern", "Good effort"],
                "areas_to_improve": ["Work on matching professional form", "Practice your timing"],
                "detailed_feedback": f"We analyzed your swing but couldn't compare it to {player_name}. Keep practicing your technique!"
            }
        
        # Update metadata
        metadata = {
            "analysis_id": analysis_id,
            "video_key": video_key,
            "status": "feedback_generated",
            "player_id": player_id,
            "frame_paths": frame_paths,
            "frame_urls": frame_urls,
            "reference_urls": reference_urls,
            "results": feedback
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        print(f"Successfully updated metadata for {analysis_id}")
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'analysis_id': analysis_id,
                'status': 'feedback_generated'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        
        # Update metadata with error status
        if analysis_id and video_key:
            try:
                error_metadata = {
                    "analysis_id": analysis_id,
                    "video_key": video_key,
                    "status": "error",
                    "error": str(e)
                }
                
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"analyses/{analysis_id}/metadata.json",
                    Body=json.dumps(error_metadata),
                    ContentType='application/json'
                )
                
                print(f"Updated metadata with error status for analysis {analysis_id}")
            except Exception as inner_e:
                print(f"Error updating metadata with error status: {str(inner_e)}")
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }

def detect_bat_region(frame):
    """Detect the bat region in a frame using edge detection and shape analysis"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    bat_region = None
    max_score = 0
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio and area
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(contour)
        rect_area = w * h
        
        # Score the contour based on bat-like properties
        # Bats typically have:
        # 1. High aspect ratio (long and thin)
        # 2. Reasonable size relative to frame
        # 3. Relatively straight edges
        frame_area = frame.shape[0] * frame.shape[1]
        size_score = area / frame_area
        shape_score = cv2.arcLength(contour, True) ** 2 / (4 * np.pi * area) if area > 0 else 0
        
        if (2.5 < aspect_ratio < 8.0 and  # Typical bat proportions
            0.01 < size_score < 0.2 and    # Reasonable size
            shape_score > 2):              # Long, straight shape
            
            score = size_score * shape_score
            if score > max_score:
                max_score = score
                bat_region = {'x': x, 'y': y, 'w': w, 'h': h}
    
    return bat_region

# Temporarily disabled batter isolation functionality
def isolate_batter(frame):
    """Isolate the batter in the frame using background subtraction and segmentation.
    
    Enhanced Implementation:
    - Traditional CV methods for batter isolation
    - Person detection using HOG descriptor
    - Background subtraction for motion-based isolation
    - Bounding box refinement
    """
    # Make a copy of the original frame
    original_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Step 1: Try to detect person using HOG descriptor (simplified person detection)
    # This is a fallback since we can't use deep learning models like YOLO in this context
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect people in the image
    boxes, weights = hog.detectMultiScale(
        frame, 
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    
    # If person detected, create a mask for the person
    person_mask = np.zeros((height, width), dtype=np.uint8)
    person_detected = False
    
    if len(boxes) > 0:
        # Find the largest box (likely the batter)
        largest_box = max(boxes, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_box
        
        # Expand the box slightly to ensure we capture the full batter
        x = max(0, x - int(w * 0.1))
        y = max(0, y - int(h * 0.1))
        w = min(width - x, int(w * 1.2))
        h = min(height - y, int(h * 1.2))
        
        # Create mask for the person
        person_mask[y:y+h, x:x+w] = 255
        person_detected = True
    
    # Step 2: Use background subtraction as an alternative approach
    # Create a simple background model using the edges of the frame
    edge_mask = np.zeros((height, width), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect components
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and position
    batter_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum size threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Check if contour is in the central region of the frame
            if (x > width * 0.2 and x + w < width * 0.8 and 
                y > height * 0.1 and y + h < height * 0.9):
                batter_contours.append(contour)
    
    # Create mask from filtered contours
    motion_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(motion_mask, batter_contours, -1, 255, -1)
    
    # Step 3: Combine approaches
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    if person_detected:
        # If person detection worked, use it as primary mask
        combined_mask = person_mask.copy()
        # Refine with motion mask where they overlap
        combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(motion_mask, person_mask))
    else:
        # Otherwise use motion mask
        combined_mask = motion_mask.copy()
    
    # Step 4: Refine the mask
    # Apply morphological operations to clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Apply the mask to isolate the batter
    # If the mask is too small or empty, return the original frame
    if np.count_nonzero(combined_mask) < (width * height * 0.05):
        # Fallback: use center region of the frame
        center_x = width // 2
        center_y = height // 2
        roi_width = width // 2
        roi_height = height // 2
        
        x1 = max(0, center_x - roi_width // 2)
        y1 = max(0, center_y - roi_height // 2)
        x2 = min(width, center_x + roi_width // 2)
        y2 = min(height, center_y + roi_height // 2)
        
        # Create a mask for the center region
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        combined_mask[y1:y2, x1:x2] = 255
    
    # Create a 3-channel mask
    mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to the original frame
    isolated_batter = cv2.bitwise_and(original_frame, mask_3channel)
    
    # For visualization, you can add a colored background
    # Create a colored background (e.g., black)
    background = np.zeros_like(original_frame)
    
    # Invert the mask for the background
    inv_mask = cv2.bitwise_not(mask_3channel)
    
    # Apply the inverted mask to the background
    masked_background = cv2.bitwise_and(background, inv_mask)
    
    # Combine the isolated batter with the background
    result = cv2.add(isolated_batter, masked_background)
    
    return result

def calculate_point_movement(prev_points, curr_points):
    """Calculate the average movement between two sets of points with normalized coordinates"""
    movements = []
    for p1, p2 in zip(prev_points, curr_points):
        if p1.get('x') is not None and p1.get('y') is not None and \
           p2.get('x') is not None and p2.get('y') is not None:
            # Calculate Euclidean distance with normalized coordinates
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            movement = np.sqrt(dx*dx + dy*dy)
            movements.append(movement)
    
    return np.mean(movements) if movements else 0.0