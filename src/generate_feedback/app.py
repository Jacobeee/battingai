import json
import boto3
import os

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

# Feedback database with common batting issues and corrections
FEEDBACK_DATABASE = {
    'stance': {
        'causes': [
            'Weight distribution is uneven',
            'Feet positioning is too narrow or too wide',
            'Improper grip on the bat'
        ],
        'corrections': [
            'Distribute weight evenly on both feet',
            'Position feet shoulder-width apart',
            'Hold the bat with a relaxed but firm grip'
        ],
        'resources': [
            {
                'type': 'youtube',
                'title': 'Perfect Baseball Stance Tutorial',
                'url': 'https://www.youtube.com/watch?v=example1'
            },
            {
                'type': 'instagram',
                'title': 'MLB Pro Stance Tips',
                'url': 'https://www.instagram.com/p/example1'
            }
        ]
    },
    'hip_rotation': {
        'causes': [
            'Hips rotating too early or too late',
            'Insufficient hip rotation',
            'Upper body moving before the hips'
        ],
        'corrections': [
            'Start hip rotation as hands begin to move forward',
            'Focus on explosive hip rotation toward the pitcher',
            'Keep upper body back until hips begin to rotate'
        ],
        'resources': [
            {
                'type': 'youtube',
                'title': 'Baseball Hip Rotation Mechanics',
                'url': 'https://www.youtube.com/watch?v=example2'
            },
            {
                'type': 'instagram',
                'title': 'Pro Hip Rotation Drills',
                'url': 'https://www.instagram.com/p/example2'
            }
        ]
    },
    'follow_through': {
        'causes': [
            'Stopping the swing too early',
            'Not extending arms fully through the swing',
            'Poor weight transfer during follow-through'
        ],
        'corrections': [
            'Complete the swing with full extension',
            'Allow the bat to naturally complete its arc',
            'Transfer weight to front foot during follow-through'
        ],
        'resources': [
            {
                'type': 'youtube',
                'title': 'Perfect Baseball Follow-Through',
                'url': 'https://www.youtube.com/watch?v=example3'
            },
            {
                'type': 'instagram',
                'title': 'MLB Follow-Through Analysis',
                'url': 'https://www.instagram.com/p/example3'
            }
        ]
    }
}

def generate_feedback(comparison_results):
    """Generate detailed feedback based on comparison results"""
    feedback = {
        'issues': [],
        'summary': ''
    }
    
    # Track unique issues to avoid duplication
    unique_issues = set()
    
    # Process each frame's issues
    for frame_result in comparison_results:
        for issue in frame_result['issues']:
            issue_type = issue['type']
            
            # Skip if we've already added this issue type
            if issue_type in unique_issues:
                continue
                
            unique_issues.add(issue_type)
            
            # Get feedback for this issue type
            if issue_type in FEEDBACK_DATABASE:
                issue_feedback = {
                    'type': issue_type,
                    'description': issue['description'],
                    'causes': FEEDBACK_DATABASE[issue_type]['causes'],
                    'corrections': FEEDBACK_DATABASE[issue_type]['corrections'],
                    'resources': FEEDBACK_DATABASE[issue_type]['resources']
                }
                feedback['issues'].append(issue_feedback)
    
    # Generate summary
    if feedback['issues']:
        issue_count = len(feedback['issues'])
        issue_types = [issue['type'] for issue in feedback['issues']]
        
        feedback['summary'] = f"Analysis identified {issue_count} key areas for improvement: {', '.join(issue_types)}. "
        feedback['summary'] += "Review the detailed feedback and watch the recommended videos to improve your technique."
    else:
        feedback['summary'] = "Your batting form looks good! No significant issues were detected in the analysis."
    
    return feedback

def lambda_handler(event, context):
    """Generate feedback based on comparison results"""
    # Define CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }
    
    try:
        # Get analysis info from event
        analysis_id = event['analysis_id']
        
        # Get comparison results
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/comparison_results.json"
        )
        comparison_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Generate feedback
        feedback = generate_feedback(comparison_data['comparison_results'])
        
        # Save feedback
        results = {
            'status': 'feedback_generated',
            'player_id': comparison_data['player_id'],
            'player_name': comparison_data['player_name'],
            'feedback': feedback
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/feedback.json",
            Body=json.dumps(results),
            ContentType='application/json'
        )
        
        # Update metadata
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=f"analyses/{analysis_id}/metadata.json"
        )
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        metadata['status'] = 'feedback_generated'
        
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
        print(f"Error generating feedback: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e)
            })
        }