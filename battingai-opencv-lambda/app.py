import json
import boto3
import os
import traceback
import tempfile
import cv2
import numpy as np
import base64  # Add base64 for encoding binary data

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

def detect_baseball(frame):
    """Detect baseball in the frame using circle detection.
    
    TODO: Improve Robustness
    - Replace hard-coded radius values (5,20) with dynamic thresholds
    - Add velocity-based validation
    - Implement ML-based ball detection
    - Add trajectory prediction
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=20
    )
    if circles is not None:
        return circles[0][0]  # Return first detected circle
    return None

def detect_bat(frame):
    """Detect bat in the frame using edge detection and line detection.
    
    TODO: Enhance Detection
    - Add ML-based bat detection
    - Implement adaptive thresholding
    - Track bat angle and speed
    - Add bat path prediction
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )
    if lines is not None:
        # Find the longest line (likely the bat)
        longest_line = max(lines, key=lambda x: np.sqrt(
            (x[0][2] - x[0][0])**2 + (x[0][3] - x[0][1])**2
        ))
        return longest_line[0]
    return None

def is_contact_frame(frame, prev_frame=None):
    """Detect if this frame shows bat-ball contact"""
    # Try to detect the baseball
    ball = detect_baseball(frame)
    if ball is None:
        return False, 0.0
    
    # Try to detect the bat
    bat = detect_bat(frame)
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

def detect_swing_phase(frames):
    """Detect the swing phases in the frames using bat-ball contact detection"""
    if len(frames) < 5:
        indices = [0]
        step = (len(frames) - 1) / 4
        for i in range(1, 4):
            indices.append(min(len(frames) - 1, int(i * step)))
        indices.append(len(frames) - 1)
        return indices
    
    # Calculate motion and contact scores for each frame
    contact_scores = []
    motion_scores = []
    prev_frame = None
    
    print("Analyzing frames for contact and motion...")
    for i, frame in enumerate(frames):
        # Check for contact
        is_contact, contact_score = is_contact_frame(frame, prev_frame)
        contact_scores.append((i, contact_score))
        
        # Calculate motion
        if prev_frame is not None:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            motion = np.mean(diff)
            motion_scores.append((i, motion))
        
        prev_frame = frame.copy()
    
    if not motion_scores:
        return [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    
    # Find contact frame (highest combined score of contact and motion)
    combined_scores = []
    for i in range(len(motion_scores)):
        _, motion = motion_scores[i]
        _, contact = contact_scores[i+1]  # offset by 1 due to motion calculation
        # Normalize scores
        motion_norm = motion / max(m for _, m in motion_scores)
        # Combine scores with higher weight on contact detection
        combined_scores.append((i+1, 0.7 * contact + 0.3 * motion_norm))
    
    # Find the frame with highest combined score - this is our contact frame
    contact_idx = max(combined_scores, key=lambda x: x[1])[0]
    
    print(f"Detected contact at frame {contact_idx}")
    
    # Now find other phases relative to contact
    total_frames = len(frames)
    
    # Setup - stable frame in first third before contact
    early_frames = list(range(max(0, contact_idx - total_frames//3), contact_idx))
    if early_frames:
        setup_idx = min(early_frames, key=lambda i: motion_scores[i][1])
    else:
        setup_idx = 0
    
    # Load - when motion starts increasing after setup
    load_window = list(range(setup_idx + 1, max(setup_idx + 2, contact_idx - 10)))
    if load_window:
        # Look for first significant motion increase
        motion_changes = [
            (i, motion_scores[i][1] - motion_scores[i-1][1])
            for i in load_window[1:]
        ]
        significant_increases = [
            (i, change) for i, change in motion_changes
            if change > np.mean([c for _, c in motion_changes]) + np.std([c for _, c in motion_changes])
        ]
        if significant_increases:
            load_idx = significant_increases[0][0]
        else:
            load_idx = setup_idx + (contact_idx - setup_idx) // 3
    else:
        load_idx = setup_idx + 1
    
    # Swing - frame just before contact showing the approach
    swing_idx = max(load_idx + 1, contact_idx - 2)
    
    # Follow-through - first clear frame after contact
    followthrough_idx = min(total_frames - 1, contact_idx + 2)
    
    # Combine phases and ensure proper ordering
    phases = [setup_idx, load_idx, swing_idx, contact_idx, followthrough_idx]
    
    # Validate frame sequence
    for i in range(1, len(phases)):
        if phases[i] <= phases[i-1]:
            phases[i] = min(total_frames - 1, phases[i-1] + 1)
    
    print(f"Detected swing phases at frames: {phases}")
    return phases

def extract_frames(video_path, num_frames=5):
    """Extract key frames from a video file.
    
    TODO: Memory Optimization
    - Replace full frame loading with streaming
    - Implement frame buffering
    - Add batch processing
    - Monitor memory usage
    - Add frame quality validation
    """
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open video file with OpenCV")
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.resize(frame, (640, 360)))
    cap.release()
    
    if len(all_frames) < 5:
        print("Not enough frames, using regular interval sampling")
        step = max(1, len(all_frames) // num_frames)
        indices = [i * step for i in range(num_frames) if i * step < len(all_frames)]
        while len(indices) < num_frames:
            indices.append(len(all_frames) - 1)
        return [all_frames[i] for i in indices[:num_frames]]
    
    print("Detecting swing phases using motion analysis...")
    
    # Use motion detection to find key swing phases
    indices = detect_swing_phase(all_frames)
    
    # Ensure we have exactly 5 frames
    if len(indices) > 5:
        indices = indices[:5]
    while len(indices) < 5:
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
    key_frames = [all_frames[i] for i in indices]
    
    print(f"Selected frame indices: {indices}")
    
    return key_frames




def save_frames_to_s3(frames, analysis_id):
    """Save extracted frames to S3.
    
    TODO: Storage Optimization
    - Add frame compression
    - Implement progressive loading
    - Add caching layer
    - Optimize metadata storage
    """
    frame_paths = []
    frame_urls = []

    print(f"Saving {len(frames)} frames to S3 for analysis {analysis_id}")
    
    for i, frame in enumerate(frames):
        try:
            # Convert frame to jpg
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Upload to S3
            frame_key = f"analyses/{analysis_id}/frames/frame_{i}.jpg"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=frame_key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )

             # Generate presigned URL
            url = get_presigned_url(frame_key)
            if url:
                frame_urls.append(url)
            
            frame_paths.append(frame_key)
            print(f"Saved frame {i} to {frame_key}")
        except Exception as e:
            print(f"Error saving frame {i}: {str(e)}")
            raise
    
    print(f"Successfully saved {len(frame_paths)} frames to S3")
    return frame_paths, frame_urls

def get_reference_frames(player_id, num_frames=5):
    """Get reference frames for the specified player"""
    reference_frames = []
    reference_urls = []

    try:
        # List reference frames for the player
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"reference/{player_id}/frames/"
        )
        
        if 'Contents' in response:
            # Sort by key to ensure frames are in order
            frame_keys = sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.jpg')])
            
            # Select evenly spaced frames if there are more than we need
            if len(frame_keys) > num_frames:
                step = len(frame_keys) // num_frames
                frame_keys = [frame_keys[i] for i in range(0, len(frame_keys), step)][:num_frames]
            
            print(f"Found {len(frame_keys)} reference frames for player {player_id}")
            
            # Download each frame
            for key in frame_keys:
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                image_data = response['Body'].read()
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Resize for consistency
                    img = cv2.resize(img, (640, 360))
                    reference_frames.append(img)
                    url = get_presigned_url(key)
                    if  url:
                        reference_urls.append(url)
                    print(f"Loaded reference frame: {key}")
                else:
                    print(f"Failed to decode reference frame: {key}")
        else:
            print(f"No reference frames found for player {player_id}")
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

def detect_swing_phase(frames):
    """Detect swing phases using advanced motion analysis.
    
    TODO: Enhance Phase Detection
    - Add ML-based phase classification
    - Implement temporal clustering
    - Add motion peak analysis
    - Improve fallback mechanism
    - Add confidence scores
    """
    if len(frames) < 5:
        return list(range(min(5, len(frames))))
    
    print("Analyzing swing mechanics...")
    
    # Get object trajectories
    obj_trajectories, bat_trajectory, ball_trajectory = detect_objects_with_background_subtraction(frames)
    
    # Analyze swing mechanics using optical flow
    motion_analysis = analyze_swing_mechanics(frames)
    
    if not motion_analysis:
        print("Failed to analyze motion, falling back to basic detection")
        return [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    
    # Calculate total frames and minimum frame spacing
    total_frames = len(frames)
    min_spacing = max(3, total_frames // 15)  # Ensure at least 3 frames between phases
    
    # First pass: Find the contact frame
    contact_idx = None
    max_magnitude = 0
    
    for i, traj in enumerate(motion_analysis):
        # Skip the first and last few frames
        if i < min_spacing or i > total_frames - min_spacing:
            continue
        if traj['magnitude'] > max_magnitude:
            max_magnitude = traj['magnitude']
            contact_idx = traj['frame_idx']
    
    if contact_idx is None or contact_idx >= total_frames - min_spacing:
        contact_idx = total_frames // 2
    
    # Refine contact frame using ball trajectory
    if ball_trajectory:
        ball_velocities = []
        for i in range(1, len(ball_trajectory)):
            dx = ball_trajectory[i][0] - ball_trajectory[i-1][0]
            dy = ball_trajectory[i][1] - ball_trajectory[i-1][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            ball_velocities.append((i-1, velocity))
        
        # Look for sudden changes in ball velocity near our initial contact estimate
        search_start = max(1, contact_idx - min_spacing)
        search_end = min(len(ball_velocities), contact_idx + min_spacing)
        
        for i in range(search_start, search_end):
            if i < len(ball_velocities):
                vel_change = ball_velocities[i][1] - ball_velocities[i-1][1]
                if vel_change > np.mean([v[1] for v in ball_velocities]) * 1.5:
                    contact_idx = ball_velocities[i][0]
                    break
    
    # Calculate other phase frames relative to contact
    setup_idx = 0
    load_idx = max(setup_idx + min_spacing, contact_idx - 3 * min_spacing)
    swing_idx = max(load_idx + min_spacing, contact_idx - 2 * min_spacing)
    followthrough_idx = min(total_frames - 1, contact_idx + 2 * min_spacing)
    
    # Ensure proper spacing between phases
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
    
    TODO: Major Enhancement Needed
    - Current: Function disabled, returns original frame
    - Replace with deep learning segmentation (Mask R-CNN/DeepLab)
    - Add person detection with YOLOv5/v8
    - Implement fallback with traditional CV methods
    - Consider transfer learning on baseball dataset
    """
    return frame