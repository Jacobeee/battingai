#!/usr/bin/env python3
import boto3
import argparse
import os
import cv2
import numpy as np
import tempfile
import json

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

def upload_reference_video(s3_client, bucket_name, video_path, player_id, player_name):
    """Upload a reference video and its frames to S3"""
    # Upload the video
    video_key = f"reference/{player_id}.mp4"
    print(f"Uploading {video_path} to s3://{bucket_name}/{video_key}")
    s3_client.upload_file(video_path, bucket_name, video_key)
    
    # Extract frames
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path)
    
    # Upload frames
    frame_paths = []
    for i, frame in enumerate(frames):
        # Convert frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Upload to S3
        frame_key = f"reference/{player_id}/frames/frame_{i}.jpg"
        print(f"Uploading frame {i} to s3://{bucket_name}/{frame_key}")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=frame_key,
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
        
        frame_paths.append(frame_key)
    
    # Create metadata
    metadata = {
        'player_id': player_id,
        'player_name': player_name,
        'video_key': video_key,
        'frame_paths': frame_paths,
        'frame_count': len(frames)
    }
    
    # Upload metadata
    metadata_key = f"reference/{player_id}/metadata.json"
    print(f"Uploading metadata to s3://{bucket_name}/{metadata_key}")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata),
        ContentType='application/json'
    )
    
    print(f"Successfully uploaded reference video for {player_name}")

def main():
    parser = argparse.ArgumentParser(description='Upload reference MLB player videos to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--player-id', required=True, help='Player ID (e.g., bryce_harper)')
    parser.add_argument('--player-name', required=True, help='Player name (e.g., Bryce Harper)')
    parser.add_argument('--profile', help='AWS profile name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    # Create S3 client
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3_client = session.client('s3')
    
    # Upload reference video
    upload_reference_video(s3_client, args.bucket, args.video, args.player_id, args.player_name)

if __name__ == '__main__':
    main()