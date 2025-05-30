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

def extract_frames(video_path):
    """Extract frames from video with proper error handling."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise Exception("Invalid frame count detected in video")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        # Sample frames at a reasonable rate to stay within Lambda timeout
        frame_interval = max(1, int(fps / 10))  # Capture up to 10 frames per second
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                
            frame_count += 1
            
        print(f"Extracted {len(frames)} frames from {frame_count} total frames")
        
    except Exception as e:
        print(f"Error in frame extraction: {str(e)}")
        raise
    finally:
        if 'cap' in locals():
            cap.release()
            
    return frames




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
        
        # Find keypoints and descriptors with mask to focus on central region
        height, width = user_gray.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center_region = [(width//4, height//4), (3*width//4, 3*height//4)]
        mask[center_region[0][1]:center_region[1][1], center_region[0][0]:center_region[1][0]] = 255
        
        user_kp, user_des = orb.detectAndCompute(user_gray, mask)
        ref_kp, ref_des = orb.detectAndCompute(ref_gray, mask)
        
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
                    match_ratio = len(good_matches) / len(matches) if len(matches) > 0 else 0
                    similarity_score = (0.6 * inliers / len(good_matches) + 0.4 * match_ratio) if len(good_matches) > 0 else 0
                else:
                    # Fallback to original distance-based scoring
                    avg_distance = sum(m.distance for m in matches) / len(matches)
                    similarity_score = max(0, min(1, 1 - (avg_distance / 100)))
            else:
                # Fallback to original distance-based scoring
                avg_distance = sum(m.distance for m in matches) / len(matches)
                similarity_score = max(0, min(1, 1 - (avg_distance / 100)))        # Calculate histogram similarity with all channels for better color matching
        hist_similarity = 0
        for channel in range(3):  # BGR channels
            hist_user = cv2.calcHist([user_frame], [channel], None, [256], [0, 256])
            cv2.normalize(hist_user, hist_user, 0, 1, cv2.NORM_MINMAX)
            
            hist_ref = cv2.calcHist([ref_frame], [channel], None, [256], [0, 256])
            cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
            
            hist_similarity += cv2.compareHist(hist_user, hist_ref, cv2.HISTCMP_CORREL) / 3

        # Calculate edge similarity with adaptive thresholds
        edges_threshold = int(np.mean(user_gray) * 0.5)
        user_edges = cv2.Canny(user_gray, 100, 200)
        ref_edges = cv2.Canny(ref_gray, 100, 200)
        
        user_edge_count = np.count_nonzero(user_edges)
        ref_edge_count = np.count_nonzero(ref_edges)
        
        edge_ratio = min(user_edge_count, ref_edge_count) / max(user_edge_count, ref_edge_count) if max(user_edge_count, ref_edge_count) > 0 else 0
        
        # Combine similarity metrics
        combined_score = (similarity_score * 0.4) + (hist_similarity * 0.3) + (edge_ratio * 0.3)
        
        # Identify issues
        issues = []
        
        if similarity_score < 0.5:
            issues.append({
                "type": "pose",
                "description": "Your body position differs from the reference"
            })
        
        if hist_similarity < 0.5:
            issues.append({
                "type": "timing",
                "description": "Your swing timing could be improved"
            })
        
        if edge_ratio < 0.5:
            issues.append({
                "type": "form",
                "description": "Your swing form needs adjustment"
            })
        
        comparison_results.append({
            "frame_index": i,
            "similarity_score": combined_score,
            "issues": issues
        })
    
    return comparison_results

def analyze_swing(user_frames, reference_frames, comparison_results):
    """Generate comprehensive swing analysis"""
    if not comparison_results:
        return {
            "overall_score": 50,
            "strengths": ["Consistent swing pattern"],
            "areas_to_improve": ["Work on matching professional form"],
            "detailed_feedback": "We couldn't perform a detailed comparison with the reference player. Try uploading a clearer video."
        }
    
    # Calculate overall score
    overall_score = int(sum(result["similarity_score"] for result in comparison_results) / len(comparison_results) * 100)
    
    # Identify common issues
    all_issues = [issue for result in comparison_results for issue in result["issues"]]
    issue_types = {}
    for issue in all_issues:
        if issue["type"] not in issue_types:
            issue_types[issue["type"]] = 0
        issue_types[issue["type"]] += 1
    
    # Determine strengths and areas to improve
    strengths = []
    areas_to_improve = []
    
    # Add strengths based on high similarity scores
    high_scores = [r for r in comparison_results if r["similarity_score"] > 0.7]
    if len(high_scores) > len(comparison_results) / 2:
        strengths.append("Good overall form matching the professional reference")
    
    if len(high_scores) > 0:
        best_frame = max(comparison_results, key=lambda x: x["similarity_score"])
        if best_frame["frame_index"] == 0:
            strengths.append("Excellent stance preparation")
        elif best_frame["frame_index"] == len(comparison_results) - 1:
            strengths.append("Strong follow-through")
        else:
            strengths.append("Good swing execution")
    
    # Add areas to improve based on issues
    if "pose" in issue_types and issue_types["pose"] > len(comparison_results) / 3:
        areas_to_improve.append("Work on your body positioning throughout the swing")
    
    if "timing" in issue_types and issue_types["timing"] > len(comparison_results) / 3:
        areas_to_improve.append("Improve your swing timing and rhythm")
    
    if "form" in issue_types and issue_types["form"] > len(comparison_results) / 3:
        areas_to_improve.append("Focus on matching the professional swing form")
    
    # Ensure we have at least one strength and area to improve
    if not strengths:
        if overall_score > 60:
            strengths.append("Consistent swing mechanics")
        else:
            strengths.append("Good effort and swing attempt")
    
    if not areas_to_improve:
        if overall_score < 80:
            areas_to_improve.append("Continue practicing to match professional form")
        else:
            areas_to_improve.append("Fine-tune your follow-through for even better results")
    
    # Generate detailed feedback
    score_text = "excellent" if overall_score > 80 else "good" if overall_score > 60 else "developing"
    detailed_feedback = f"Your swing shows {score_text} potential with a similarity score of {overall_score}/100 compared to the reference player. "
    
    if len(strengths) > 0:
        detailed_feedback += f"Your key strength is {strengths[0].lower()}. "
    
    if len(areas_to_improve) > 0:
        detailed_feedback += f"To improve, focus on {areas_to_improve[0].lower()}."
    
    return {
        "overall_score": overall_score,
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "detailed_feedback": detailed_feedback,
        "comparison_results": comparison_results
    }

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
    """Detect swing phases using advanced motion analysis and object tracking"""
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
            "player_id": player_id,
            "player_name": player_name
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
                print(f"Downloading video from S3: {video_key}")
                s3_client.download_file(bucket_name, video_key, temp_path)
                
                # Extract frames from video
                print("Extracting frames from video...")
                frames = extract_frames(temp_path)
                
                if not frames or len(frames) == 0:
                    raise Exception("Failed to extract frames from video")
                    
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                metadata.update({
                    "status": "failed",
                    "error": f"Failed to process video: {str(e)}"
                })
                # Update metadata with error
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"analyses/{analysis_id}/metadata.json",
                    Body=json.dumps(metadata),
                    ContentType='application/json'
                )
                raise
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp file {temp_path}: {str(e)}")

        # Save frames to S3
        frame_paths, frame_urls = save_frames_to_s3(frames, analysis_id)
        
        # Get reference frames for the selected player
        reference_frames, reference_urls = get_reference_frames(player_id)
        
        # Compare user frames with reference frames
        comparison_results = compare_frames(frames, reference_frames)
          # Generate comprehensive analysis and update metadata
        if comparison_results:
            analysis_results = analyze_swing(frames, reference_frames, comparison_results)
            metadata.update({
                "status": "completed",
                "comparison_results": comparison_results,
                "analysis_results": analysis_results,
                "frame_paths": frame_paths,
                "frame_urls": frame_urls,
                "reference_urls": reference_urls,
                "feedback": (
                    f"Based on your {analysis_results['overall_score']}/100 score comparing to {player_name}:\n\n"
                    f"Strengths:\n- " + "\n- ".join(analysis_results['strengths']) + "\n\n"
                    f"Areas to Improve:\n- " + "\n- ".join(analysis_results['areas_to_improve']) + "\n\n"
                    f"{analysis_results['detailed_feedback']}"
                )
            })
            
            # Return successful response immediately
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/metadata.json",
                Body=json.dumps(metadata),
                ContentType='application/json'
            )
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(metadata)
            }
        else:
            # Handle case where comparison failed
            metadata.update({
                "status": "completed",
                "error": "Unable to perform detailed comparison",
                "frame_paths": frame_paths,
                "frame_urls": frame_urls,
                "reference_urls": reference_urls,
                "feedback": (
                    f"We've analyzed your swing but couldn't perform a detailed comparison with {player_name}. "
                    "This could be due to lighting conditions, video quality, or camera angle. "
                    "For best results, try recording:\n"
                    "- In good lighting conditions\n"
                    "- With a stable camera\n"
                    "- From a side view\n"
                    "- With the full swing motion visible"
                )
            })        # Save metadata and return response for failed comparison case
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(metadata)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
          # Update metadata with error status
        if analysis_id and video_key:
            try:
                metadata.update({
                    "status": "error",
                    "error": str(e),
                    "feedback": (
                        "We encountered an error while analyzing your swing. "
                        "This could be due to:\n"
                        "- Video format or quality issues\n"
                        "- Processing limitations\n"
                        "- System errors\n\n"
                        "Please try uploading your video again or contact support if the issue persists."
                    )
                })
                
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"analyses/{analysis_id}/metadata.json",
                    Body=json.dumps(metadata),
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