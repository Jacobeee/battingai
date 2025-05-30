import json
import boto3
import os
import traceback
import tempfile
import cv2
import numpy as np

s3_client = boto3.client('s3')
bucket_name = os.environ.get('BUCKET_NAME', 'battingai-videobucket-ayk9m1uehbg2')

# Player name mapping
PLAYER_NAMES = {
    'bryce_harper': 'Bryce Harper',
    'brandon_lowe': 'Brandon Lowe',
    'mike_trout': 'Mike Trout',
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
    """Detect baseball in the frame using circle detection"""
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
    """Detect bat in the frame using edge detection and line detection"""
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
    """Extract key frames from a video file based on fixed percentages"""
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
    """Save extracted frames to S3"""
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
    """Compare user frames with reference frames using advanced image processing"""
    if not reference_frames:
        print("No reference frames available for comparison")
        return None
    
    # Ensure we have the same number of frames to compare
    min_frames = min(len(user_frames), len(reference_frames))
    user_frames = user_frames[:min_frames]
    reference_frames = reference_frames[:min_frames]
    
    comparison_results = []
    
    for i, (user_frame, ref_frame) in enumerate(zip(user_frames, reference_frames)):
        # Normalize lighting
        user_lab = cv2.cvtColor(user_frame, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2LAB)
        
        # Normalize L channel (lighting)
        user_l = cv2.normalize(user_lab[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        ref_l = cv2.normalize(ref_lab[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        
        # Reconstruct normalized images
        user_lab[:,:,0] = user_l
        ref_lab[:,:,0] = ref_l
        user_norm = cv2.cvtColor(user_lab, cv2.COLOR_LAB2BGR)
        ref_norm = cv2.cvtColor(ref_lab, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale for feature detection
        user_gray = cv2.cvtColor(user_norm, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_norm, cv2.COLOR_BGR2GRAY)
        
        # Initialize feature detector with better parameters for baseball swings
        orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31
        )
        
        # Find keypoints and descriptors
        user_kp, user_des = orb.detectAndCompute(user_gray, None)
        ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
        
        # Match features and calculate similarity
        similarity_score = 0
        if user_des is not None and ref_des is not None and len(user_des) > 0 and len(ref_des) > 0:
            # Use better feature matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(user_des, ref_des)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use best 70% of matches for homography
            good_matches = matches[:int(len(matches) * 0.7)]
            
            if len(good_matches) >= 4:
                # Get matched keypoints
                src_pts = np.float32([user_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Calculate similarity score based on good matches ratio and homography quality
                    inliers = np.sum(mask)
                    match_ratio = len(good_matches) / len(matches) if len(matches) > 0