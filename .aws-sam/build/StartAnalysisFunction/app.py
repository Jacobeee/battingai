import json
import boto3
import os
import traceback

lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')
bucket_name = "battingai-videobucket-ayk9m1uehbg2"  # Hardcoded bucket name since environment variable is missing

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
            print("Error: No body in event")
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Invalid request format'
                })
            }
        
        body = json.loads(event['body'])
        print(f"Parsed body: {json.dumps(body)}")
        
        # Get analysis ID
        analysis_id = body.get('analysis_id')
        if not analysis_id:
            print("Error: No analysis_id in body")
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'No analysis_id provided'
                })
            }
        
        print(f"Looking for analysis with ID: {analysis_id}")
        
        # Get player ID for comparison (optional)
        player_id = body.get('player_id', 'bryce_harper')  # Default to Bryce Harper
        print(f"Using player_id: {player_id}")
        
        # Get metadata for the analysis
        try:
            metadata_key = f"analyses/{analysis_id}/metadata.json"
            print(f"Looking for metadata at: {bucket_name}/{metadata_key}")
            
            # List objects in the analyses directory to debug
            try:
                list_response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=f"analyses/{analysis_id}/"
                )
                if 'Contents' in list_response:
                    print(f"Found objects in analyses/{analysis_id}/: {[obj['Key'] for obj in list_response['Contents']]}")
                else:
                    print(f"No objects found in analyses/{analysis_id}/")
            except Exception as list_error:
                print(f"Error listing objects: {str(list_error)}")
            
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=metadata_key
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            print(f"Found metadata: {json.dumps(metadata)}")
        except Exception as e:
            print(f"Error getting metadata: {str(e)}")
            print(traceback.format_exc())
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Analysis not found'
                })
            }
        
        # Check if video has been uploaded - accept both 'uploaded' and 'pending' status
        # 'pending' is the new status used by the presigned URL workflow
        print(f"Current metadata status: {metadata['status']}")
        if metadata['status'] not in ['uploaded', 'pending']:
            print(f"Invalid status: {metadata['status']}")
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
        
        print(f"Updating metadata: {json.dumps(metadata)}")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        # Start the processing workflow
        # 1. Process video to extract frames
        process_video_function = os.environ.get('PROCESS_VIDEO_FUNCTION', '')
        print(f"Process video function ARN: {process_video_function}")
        
        if process_video_function:
            try:
                payload = {
                    'analysis_id': analysis_id,
                    'video_key': metadata['video_key']
                }
                print(f"Invoking process video function with payload: {json.dumps(payload)}")
                
                process_video_response = lambda_client.invoke(
                    FunctionName=process_video_function,
                    InvocationType='Event',  # Asynchronous
                    Payload=json.dumps(payload)
                )
                print(f"Successfully invoked ProcessVideoFunction: {process_video_function}")
                print(f"Response: {process_video_response}")
            except Exception as e:
                print(f"Error invoking ProcessVideoFunction: {str(e)}")
                print(f"Function ARN: {process_video_function}")
                print(traceback.format_exc())
                raise
        else:
            print("No process video function ARN provided")
        
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