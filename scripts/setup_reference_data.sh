#!/bin/bash
# Script to set up reference data in the S3 bucket

# Get the stack name from samconfig.toml
STACK_NAME=$(grep stack_name samconfig.toml | cut -d'"' -f2)

# Get the S3 bucket name from CloudFormation outputs
BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='VideoBucket'].OutputValue" --output text)

echo "Using S3 bucket: $BUCKET_NAME"

# Create directories for reference videos
mkdir -p reference_videos

# Download reference videos (replace with actual URLs)
echo "Downloading reference videos..."
# Example: wget -O reference_videos/bryce_harper.mp4 https://example.com/bryce_harper_batting.mp4
# Example: wget -O reference_videos/brandon_lowe.mp4 https://example.com/brandon_lowe_batting.mp4
wget -O reference_videos/bryce_harper.mp4 C:\Users\jcewa\Documents\battingai\bryce.mp4
wget -O reference_videos/brandon_lowe.mp4 C:\Users\jcewa\Documents\battingai\bryce.mp4 #temp

# Upload reference videos using the updated script that uses the same functionality as user videos
echo "Uploading reference videos..."
python scripts/upload_reference_videos.py --bucket $BUCKET_NAME --video reference_videos/bryce_harper.mp4 --player-id bryce_harper --player-name "Bryce Harper"
if [ $? -ne 0 ]; then
    echo "Failed to upload Bryce Harper reference video"
    exit 1
fi

python scripts/upload_reference_videos.py --bucket $BUCKET_NAME --video reference_videos/brandon_lowe.mp4 --player-id brandon_lowe --player-name "Brandon Lowe"
if [ $? -ne 0 ]; then
    echo "Failed to upload Brandon Lowe reference video"
    exit 1
fi

echo "Reference data setup complete!"