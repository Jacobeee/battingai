import json
import boto3
import os
import cv2
import numpy as np
from io import BytesIO
import tempfile

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

# Reference MLB players data
MLB_PLAYERS = {
    'bryce_harper': {
        'name': 'Bryce Harper',
        'video_key': 'reference/bryce_harper.mp4',
        'frames_key': 'reference/bryce_harper/frames/'
    },
    'brandon_lowe': {
        'name': 'Brandon Lowe',
        'video_key': 'reference/brandon_lowe.mp4',
        'frames_key': 'reference/brandon_lowe/frames/'
    }
}

def download_frames(frame_paths):
    """Download frames from S3"""
    frames = []
    
    for path in frame_paths:
        response = s3_client.get_object(Bucket=bucket_name, Key=path)
        img_bytes = response['Body'].read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(img)
    
    return frames

def download_reference_frames(player_id):
    """Download reference frames for an MLB player"""
    frames = []
    player = MLB_PLAYERS[player_id]
    
    # List objects in the reference frames directory
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=player['frames_key']
    )
    
    if 'Contents' in response:
        frame_paths = [obj['Key'] for obj in response['Contents']]
        frames = download_frames(frame_paths)
    
    return frames

def compare_frames(user_frames, reference_frames):
    """Compare user frames with reference frames using pose estimation and similarity metrics"""
    # This is a simplified comparison - in a real application, you would use 
    # more sophisticated pose estimation and comparison algorithms
    
    results = []
    
    # Define swing phases
    phases = ['setup', 'load', 'swing', 'contact', 'follow-through']
    
    # Ensure we have 5 frames from each video (one per phase)
    if len(user_frames) != 5 or len(reference_frames) != 5:
        print(f"Warning: Expected 5 frames per video, got {len(user_frames)} and {len(reference_frames)}")
    
    # Compare corresponding frames from each phase
    for i, phase in enumerate(phases):
        if i >= len(user_frames) or i >= len(reference_frames):
            break
            
        user_frame = user_frames[i]
        ref_frame = reference_frames[i]
        
        # Calculate histogram similarity
        user_hist = cv2.calcHist([user_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        ref_hist = cv2.calcHist([ref_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(user_hist, user_hist)
        cv2.normalize(ref_hist, ref_hist)
        
        # Compare histograms
        similarity = cv2.compareHist(user_hist, ref_hist, cv2.HISTCMP_CORREL)
        
        # For identical videos, similarity should be very close to 1.0
        adjusted_similarity = similarity * 100  # Convert to percentage
        
        # Basic image analysis for stance detection
        user_height, user_width = user_frame.shape[:2]
        ref_height, ref_width = ref_frame.shape[:2]
        
        # Calculate lower half of image (where feet would be)
        user_lower_half = user_frame[user_height//2:, :]
        ref_lower_half = ref_frame[ref_height//2:, :]
        
        # Convert to grayscale for edge detection
        user_gray = cv2.cvtColor(user_lower_half, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_lower_half, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        user_edges = cv2.Canny(user_gray, 100, 200)
        ref_edges = cv2.Canny(ref_gray, 100, 200)
        
        # Count edge pixels as a simple measure of stance width
        user_edge_count = np.count_nonzero(user_edges)
        ref_edge_count = np.count_nonzero(ref_edges)
        
        # Compare stance widths
        stance_diff = abs(user_edge_count - ref_edge_count) / max(user_edge_count, ref_edge_count)
        stance_similarity = 100 * (1 - stance_diff)
        
        # Combined score (weighted average of histogram and stance similarities)
        if phase == 'setup':
            # For setup phase, stance is more important
            final_score = 0.3 * adjusted_similarity + 0.7 * stance_similarity
        else:
            # For other phases, overall position (histogram) is more important
            final_score = 0.7 * adjusted_similarity + 0.3 * stance_similarity
        
        # Analyze specific issues based on the phase
        issues = []
        if final_score < 90:  # Threshold for identifying issues
            if phase == 'setup':
                if stance_similarity < 90:
                    issues.append({
                        'type': 'stance',
                        'description': 'Your stance width differs from the reference'
                    })
            elif phase == 'load':
                if adjusted_similarity < 90:
                    issues.append({
                        'type': 'loading',
                        'description': 'Your loading position differs from the reference'
                    })
            elif phase == 'swing':
                if adjusted_similarity < 90:
                    issues.append({
                        'type': 'swing_path',
                        'description': 'Your swing path differs from the reference'
                    })
        
        results.append({
            'frame_index': i,
            'phase': phase,
            'similarity_score': round(final_score, 2),
            'issues': issues
        })
    
    return results

def lambda_handler(event, context):
    """Compare user video frames with reference MLB player frames"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With,Accept,Origin',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
    }
    
    # Handle OPTIONS request (preflight)
    if event.get('httpMethod') == 'OPTIONS':
        print(f"Handling OPTIONS request with headers: {event.get('headers', {})}")
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Get analysis info from event
        analysis_id = event['analysis_id']
        player_id = event.get('player_id', 'bryce_harper')  # Default to Bryce Harper
        
        # Get metadata for the analysis
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json"
        )
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        # Get user frames
        user_frame_paths = metadata['frame_paths']
        user_frames = download_frames(user_frame_paths)
        
        # Get reference frames
        reference_frames = download_reference_frames(player_id)
        
        # Compare frames
        comparison_results = compare_frames(user_frames, reference_frames)
        
        # Save comparison results
        results = {
            'status': 'comparison_complete',
            'player_id': player_id,
            'player_name': MLB_PLAYERS[player_id]['name'],
            'comparison_results': comparison_results
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/comparison_results.json",
            Body=json.dumps(results),
            ContentType='application/json'
        )
        
        # Update metadata
        metadata['status'] = 'comparison_complete'
        metadata['player_id'] = player_id
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'analysis_id': analysis_id,
                'status': 'comparison_complete',
                'player_id': player_id
            })
        }
        
    except Exception as e:
        print(f"Error comparing videos: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }