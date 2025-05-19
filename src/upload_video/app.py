import json
import boto3
import os
import base64
import uuid
from datetime import datetime

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

def lambda_handler(event, context):
    """Handle video upload from client"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With,Accept',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }
    
    # Handle OPTIONS request (preflight)
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Parse request body
        body = json.loads(event['body'])
        
        # Get video data (base64 encoded)
        video_data = body.get('video')
        if not video_data:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'No video data provided'
                })
            }
        
        # Decode base64 video data
        video_bytes = base64.b64decode(video_data)
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        video_key = f"uploads/{timestamp}_{analysis_id}.mp4"
        
        # Upload video to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=video_key,
            Body=video_bytes,
            ContentType='video/mp4'
        )
        
        # Create initial metadata
        metadata = {
            'analysis_id': analysis_id,
            'video_key': video_key,
            'timestamp': timestamp,
            'status': 'uploaded'
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
                'status': 'uploaded'
            })
        }
        
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }