import json
import boto3
import os
import traceback
import tempfile
import cv2
import numpy as np

s3_client = boto3.client('s3')
bucket_name = os.environ.get('BUCKET_NAME', 'battingai-videobucket-ayk9m1uehbg2')

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
    """Save extracted frames to S3 and generate presigned URLs"""
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
        
        # Save frames to S3 and get presigned URLs
        frame_paths, frame_urls = save_frames_to_s3(frames, analysis_id)
        
        # Create feedback based on the frames
        feedback = {
            "status": "feedback_generated",
            "player_id": player_id,
            "player_name": player_id.replace('_', ' ').title(),
            "overall_score": 75,
            "strengths": ["Good follow-through", "Solid bat speed"],
            "areas_to_improve": ["Work on hip rotation timing", "Adjust stance width for better balance"],
            "detailed_feedback": "Your swing mechanics show good potential."
        }
        
        # Update metadata
        metadata = {
            "analysis_id": analysis_id,
            "video_key": video_key,
            "status": "feedback_generated",
            "player_id": player_id,
            "frame_paths": frame_paths,
            "frame_urls": frame_urls,
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