import json
import boto3
import os
import cv2
import numpy as np
from io import BytesIO
import tempfile

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

def extract_frames(video_path, num_frames=10):
    """Extract key frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (evenly distributed)
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def save_frames_to_s3(frames, analysis_id):
    """Save extracted frames to S3"""
    frame_paths = []
    
    for i, frame in enumerate(frames):
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
        
        frame_paths.append(frame_key)
    
    return frame_paths

def lambda_handler(event, context):
    """Process video to extract frames for analysis"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }
    
    try:
        # Get video info from event
        video_key = event['video_key']
        analysis_id = event['analysis_id']
        
        # Download video from S3
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            s3_client.download_file(bucket_name, video_key, temp_video.name)
            
            # Extract frames
            frames = extract_frames(temp_video.name)
            
            # Save frames to S3
            frame_paths = save_frames_to_s3(frames, analysis_id)
            
            # Update analysis metadata
            metadata = {
                'status': 'frames_extracted',
                'video_key': video_key,
                'frame_paths': frame_paths,
                'frame_count': len(frames)
            }
            
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
                    'status': 'frames_extracted',
                    'frame_count': len(frames)
                })
            }
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }