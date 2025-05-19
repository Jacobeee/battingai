import json
import boto3
import os

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

def lambda_handler(event, context):
    """Get analysis results for a specific analysis ID"""
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
        # Get analysis ID from path parameter
        analysis_id = event['pathParameters']['analysisId']
        
        # Get metadata for the analysis
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/metadata.json"
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Analysis not found'
                })
            }
        
        # Check if feedback has been generated
        if metadata['status'] == 'feedback_generated':
            # Get feedback
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/feedback.json"
            )
            feedback = json.loads(response['Body'].read().decode('utf-8'))
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'analysis_id': analysis_id,
                    'status': metadata['status'],
                    'results': feedback
                })
            }
        elif metadata['status'] == 'comparison_complete':
            # Get comparison results
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"analyses/{analysis_id}/comparison_results.json"
            )
            comparison = json.loads(response['Body'].read().decode('utf-8'))
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'analysis_id': analysis_id,
                    'status': metadata['status'],
                    'results': comparison
                })
            }
        else:
            # Analysis is still in progress
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'analysis_id': analysis_id,
                    'status': metadata['status']
                })
            }
        
    except Exception as e:
        print(f"Error getting results: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }