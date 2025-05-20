import json
import boto3
import os
import traceback

lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')
bucket_name = os.environ.get('BUCKET_NAME', '')

def lambda_handler(event, context):
    """Start the analysis workflow for a video"""
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
        # Parse request body
        print(f"Received event: {json.dumps(event)}")
        
        if 'body' not in event:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Invalid request format'
                })
            }
        
        body = json.loads(event['body'])
        
        # Get analysis ID
        analysis_id = body.get('analysis_id')
        if not analysis_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'No analysis_id provided'
                })
            }
        
        # Get player ID for comparison (optional)
        player_id = body.get('player_id', 'bryce_harper')  # Default to Bryce Harper
        
        # Get metadata for the analysis
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/metadata.json"
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error getting metadata: {str(e)}")
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Analysis not found'
                })
            }
        
        # Check if video has been uploaded
        if metadata['status'] != 'uploaded':
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': f"Invalid analysis status: {metadata['status']}"
                })
            }
        
        # Update metadata
        metadata['status'] = 'processing'
        metadata['player_id'] = player_id
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        # Start the processing workflow
        # 1. Process video to extract frames
        process_video_function = os.environ.get('PROCESS_VIDEO_FUNCTION', '')
        if process_video_function:
            try:
                process_video_response = lambda_client.invoke(
                    FunctionName=process_video_function,
                    InvocationType='Event',  # Asynchronous
                    Payload=json.dumps({
                        'analysis_id': analysis_id,
                        'video_key': metadata['video_key']
                    })
                )
                print(f"Successfully invoked ProcessVideoFunction: {process_video_function}")
            except Exception as e:
                print(f"Error invoking ProcessVideoFunction: {str(e)}")
                print(f"Function ARN: {process_video_function}")
                raise
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'analysis_id': analysis_id,
                'status': 'processing',
                'player_id': player_id
            })
        }
        
    except Exception as e:
        print(f"Error starting analysis: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }