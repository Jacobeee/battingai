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

def extract_frames(video_path, num_frames=5):
    """Extract key frames from a video file"""
    print(f"Opening video file: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    print(f"File size: {os.path.getsize(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open video file with OpenCV")
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: frames={frame_count}, fps={fps}, resolution={width}x{height}")
    
    if frame_count <= 0:
        print("Invalid frame count, trying to read frames directly")
        # Try to read frames directly
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)
                print(f"Read frame {i} directly")
            else:
                print(f"Failed to read frame {i} directly")
                break
        
        cap.release()
        
        if not frames:
            raise Exception("No frames could be extracted from the video")
        
        return frames
    
    # Extract frames at regular intervals
    frames = []
    step = max(1, frame_count // num_frames)
    
    for i in range(0, min(frame_count, num_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
            print(f"Extracted frame at position {i}")
        else:
            print(f"Failed to read frame at position {i}")
    
    cap.release()
    
    if not frames:
        raise Exception("No frames could be extracted from the video")
    
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
    """Compare user frames with reference frames"""
    if not reference_frames:
        print("No reference frames available for comparison")
        return None
    
    # Ensure we have the same number of frames to compare
    min_frames = min(len(user_frames), len(reference_frames))
    user_frames = user_frames[:min_frames]
    reference_frames = reference_frames[:min_frames]
    
    comparison_results = []
    
    for i, (user_frame, ref_frame) in enumerate(zip(user_frames, reference_frames)):
        # Convert to grayscale for feature detection
        user_gray = cv2.cvtColor(user_frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize feature detector
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        user_kp, user_des = orb.detectAndCompute(user_gray, None)
        ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
        
        # Match features
        similarity_score = 0
        if user_des is not None and ref_des is not None and len(user_des) > 0 and len(ref_des) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(user_des, ref_des)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity score (0-1)
            if len(matches) > 0:
                avg_distance = sum(m.distance for m in matches) / len(matches)
                similarity_score = max(0, min(1, 1 - (avg_distance / 100)))
        
        # Calculate histogram similarity
        hist_user = cv2.calcHist([user_gray], [0], None, [256], [0, 256])
        hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
        
        cv2.normalize(hist_user, hist_user, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
        
        hist_similarity = cv2.compareHist(hist_user, hist_ref, cv2.HISTCMP_CORREL)
        
        # Calculate edge similarity
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