import json
import boto3
import os
import traceback

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
bucket_name = os.environ.get('BUCKET_NAME', 'battingai-videobucket-ayk9m1uehbg2')

def lambda_handler(event, context):
    """Process video to extract frames for analysis"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': 'https://jacobeee.github.io',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With,Accept,Origin',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
    }
    
    try:
        # Get video info from event
        print(f"Received event: {json.dumps(event)}")
        video_key = event['video_key']
        analysis_id = event['analysis_id']
        print(f"Processing video: {video_key} for analysis: {analysis_id}")
        
        # Skip actual video processing and create mock data
        
        # 1. Create mock frame paths
        frame_paths = [f"analyses/{analysis_id}/frames/frame_{i}.jpg" for i in range(5)]
        
        # 2. Create mock comparison results
        comparison_results = {
            "status": "comparison_complete",
            "player_id": "bryce_harper",
            "player_name": "Bryce Harper",
            "comparison_results": [
                {
                    "frame_index": 0,
                    "similarity_score": 0.85,
                    "issues": []
                },
                {
                    "frame_index": 1,
                    "similarity_score": 0.72,
                    "issues": [
                        {
                            "type": "stance",
                            "description": "Your stance is slightly wider than the reference"
                        }
                    ]
                },
                {
                    "frame_index": 2,
                    "similarity_score": 0.68,
                    "issues": [
                        {
                            "type": "hip_rotation",
                            "description": "Your hip rotation could be more pronounced"
                        }
                    ]
                }
            ]
        }
        
        # 3. Create mock feedback
        feedback = {
            "status": "feedback_generated",
            "player_id": "bryce_harper",
            "player_name": "Bryce Harper",
            "overall_score": 75,
            "strengths": [
                "Good follow-through",
                "Solid bat speed"
            ],
            "areas_to_improve": [
                "Work on hip rotation timing",
                "Adjust stance width for better balance"
            ],
            "detailed_feedback": "Your swing mechanics show good potential. Focus on improving your hip rotation timing to generate more power. Your stance could be adjusted slightly for better balance throughout your swing."
        }
        
        # 4. Upload the mock data to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/comparison_results.json",
            Body=json.dumps(comparison_results),
            ContentType='application/json'
        )
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/feedback.json",
            Body=json.dumps(feedback),
            ContentType='application/json'
        )
        
        # 5. Update metadata to feedback_generated
        metadata = {
            "analysis_id": analysis_id,
            "video_key": video_key,
            "status": "feedback_generated",
            "player_id": "bryce_harper",
            "frame_paths": frame_paths
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
                'status': 'feedback_generated'
            })
        }
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }
