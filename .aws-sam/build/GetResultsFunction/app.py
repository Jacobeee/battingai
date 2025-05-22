import json
import boto3
import os

s3_client = boto3.client('s3')
bucket_name = os.environ.get('BUCKET_NAME', 'battingai-videobucket-ayk9m1uehbg2')

def lambda_handler(event, context):
    """Get analysis results"""
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With,Accept,Origin',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
    }
    
    try:
        # Get analysis ID from path parameters
        analysis_id = event.get('pathParameters', {}).get('analysis_id')
        if not analysis_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Missing analysis_id parameter'
                })
            }
        
        print(f"Getting results for analysis: {analysis_id}")
        
        # Try to get the metadata file
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/metadata.json"
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            print(f"Found metadata for analysis: {analysis_id}")
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(metadata)
            }
        except s3_client.exceptions.NoSuchKey:
            # Metadata file doesn't exist yet, return processing status
            print(f"Metadata not found for analysis: {analysis_id}")
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'analysis_id': analysis_id,
                    'status': 'processing'
                })
            }
        
    except Exception as e:
        print(f"Error getting results: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }
