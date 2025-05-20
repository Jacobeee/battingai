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
    
    # Ensure we have the same number of frames to compare
    min_frames = min(len(user_frames), len(reference_frames))
    
    for i in range(min_frames):
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
        
        # In a real application, you would use pose estimation to compare specific body positions
        # For this example, we'll use a simplified approach
        
        results.append({
            'frame_index': i,
            'similarity_score': float(similarity),
            'issues': []
        })
        
        # Add mock issues based on similarity score
        if similarity < 0.5:
            if i < 3:  # Early frames - setup and stance
                results[i]['issues'].append({
                    'type': 'stance',
                    'description': 'Your stance is too narrow compared to the reference'
                })
            elif i < 6:  # Middle frames - swing initiation
                results[i]['issues'].append({
                    'type': 'hip_rotation',
                    'description': 'Your hip rotation is delayed compared to the reference'
                })
            else:  # Late frames - follow through
                results[i]['issues'].append({
                    'type': 'follow_through',
                    'description': 'Your follow-through is incomplete compared to the reference'
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