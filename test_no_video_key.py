import json
import sys
import os
sys.path.append('/workspace/battingai-opencv-lambda')
from app import lambda_handler

# Test event with no video_key
test_event = {
    'body': json.dumps({
        'analysis_id': 'test-analysis-id',
        'player_id': 'bryce_harper'
    })
}

# Mock context
class MockContext:
    def __init__(self):
        self.function_name = 'test-function'
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
        self.aws_request_id = 'test-request-id'

# Call the lambda handler
print("Testing lambda_handler with no video_key...")
response = lambda_handler(test_event, MockContext())

# Print the response
print(f"Response status code: {response['statusCode']}")
print(f"Response body: {response['body']}")

# Test event with video_key
test_event_with_video = {
    'body': json.dumps({
        'analysis_id': 'test-analysis-id-with-video',
        'player_id': 'bryce_harper',
        'video_key': 'test-video-key'
    })
}

# Call the lambda handler with video_key
print("\nTesting lambda_handler with video_key...")
try:
    response = lambda_handler(test_event_with_video, MockContext())
    print(f"Response status code: {response['statusCode']}")
    print(f"Response body: {response['body']}")
except Exception as e:
    print(f"Error: {str(e)}")
    print("This is expected if the video file doesn't exist in S3")