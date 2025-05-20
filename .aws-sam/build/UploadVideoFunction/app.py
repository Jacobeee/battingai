import json
import boto3
import os
import uuid
from datetime import datetime

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

def lambda_handler(event, context):
    """Generate presigned URL for direct S3 upload"""
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
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        video_key = f"uploads/{timestamp}_{analysis_id}.mp4"
        
        # Generate presigned URL for S3 upload
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': video_key,
                'ContentType': 'video/mp4'
            },
            ExpiresIn=300  # URL valid for 5 minutes
        )
        
        # Create initial metadata
        metadata = {
            'analysis_id': analysis_id,
            'video_key': video_key,
            'timestamp': timestamp,
            'status': 'pending'  # Will be updated to 'uploaded' after successful upload
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
                'upload_url': presigned_url,
                'video_key': video_key
            })
        }
        
    except Exception as e:
        print(f"Error generating upload URL: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }