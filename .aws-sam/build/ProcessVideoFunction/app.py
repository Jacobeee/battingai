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
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: frames={frame_count}, fps={fps}, resolution={width}x{height}")
    
    if frame_count <= 0:
        raise Exception(f"Invalid frame count: {frame_count}")
    
    # Calculate frame indices to extract (evenly distributed)
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    print(f"Extracting frames at indices: {indices}")
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to read frame at index {idx}")
    
    print(f"Successfully extracted {len(frames)} frames out of {num_frames} requested")
    cap.release()
    return frames

def save_frames_to_s3(frames, analysis_id):
    """Save extracted frames to S3"""
    frame_paths = []
    
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
            
            frame_paths.append(frame_key)
            print(f"Saved frame {i} to {frame_key}")
        except Exception as e:
            print(f"Error saving frame {i}: {str(e)}")
            raise
    
    print(f"Successfully saved {len(frame_paths)} frames to S3")
    return frame_paths

def lambda_handler(event, context):
    """Process video to extract frames for analysis"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,Accept,Origin',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
    }
    
    print(f"ProcessVideoFunction invoked with event: {json.dumps(event)}")
    print(f"Using bucket: {bucket_name}")
    
    try:
        # Get video info from event
        print(f"Received event: {json.dumps(event)}")
        video_key = event['video_key']
        analysis_id = event['analysis_id']
        print(f"Processing video: {video_key} for analysis: {analysis_id}")
        
        # Download video from S3
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_path = temp_video.name
            print(f"Downloading video from S3: {bucket_name}/{video_key} to {temp_path}")
            
            try:
                s3_client.download_file(bucket_name, video_key, temp_path)
                print(f"Successfully downloaded video to {temp_path}")
                
                # Check if file exists and has content
                import os
                file_size = os.path.getsize(temp_path)
                print(f"Downloaded file size: {file_size} bytes")
                
                if file_size == 0:
                    raise Exception(f"Downloaded file is empty: {temp_path}")
                
                # Extract frames
                frames = extract_frames(temp_path)
            except Exception as e:
                print(f"Error during video download or frame extraction: {str(e)}")
                raise
            finally:
                # Clean up the temp file
                try:
                    os.unlink(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error cleaning up temp file: {str(e)}")
            
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
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {error_trace}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e),
                'traceback': error_trace
            })
        }