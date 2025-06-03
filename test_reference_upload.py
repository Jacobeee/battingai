#!/usr/bin/env python3
"""
Test script to verify that reference videos are processed correctly
using the same functionality as user videos.
"""
import os
import sys
import boto3
import json
import tempfile
import cv2
import numpy as np
import time
import traceback

# Add the parent directory to the Python path to import from app.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'battingai-opencv-lambda'))

# Import functions from app.py
try:
    from app import extract_frames, save_frames_to_s3, get_reference_frames
except ImportError as e:
    print(f"Error importing functions from app.py: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import the upload_reference_video function
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from upload_reference_videos import upload_reference_video
except ImportError as e:
    print(f"Error importing functions from upload_reference_videos.py: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_reference_upload(bucket_name, video_path, player_id="test_player", player_name="Test Player"):
    """Test uploading a reference video and retrieving its frames"""
    print(f"Testing reference video upload with {video_path}")
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Upload reference video
    success = upload_reference_video(s3_client, bucket_name, video_path, player_id, player_name)
    
    if not success:
        print("Failed to upload reference video")
        return False
    
    print("Reference video uploaded successfully")
    
    # Get reference frames
    reference_frames, reference_urls = get_reference_frames(player_id)
    
    if not reference_frames:
        print("Failed to retrieve reference frames")
        return False
    
    print(f"Successfully retrieved {len(reference_frames)} reference frames")
    print(f"Reference URLs: {reference_urls}")
    
    # Display the first frame (if running in an environment with display)
    try:
        cv2.imshow("Reference Frame", reference_frames[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display frame (no display available)")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test reference video upload and retrieval')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--player-id', default='test_player', help='Player ID (e.g., test_player)')
    parser.add_argument('--player-name', default='Test Player', help='Player name (e.g., Test Player)')
    
    args = parser.parse_args()
    
    if test_reference_upload(args.bucket, args.video, args.player_id, args.player_name):
        print("Test completed successfully")
        sys.exit(0)
    else:
        print("Test failed")
        sys.exit(1)