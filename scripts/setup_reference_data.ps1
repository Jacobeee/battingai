# PowerShell script to set up reference data
# Fix paths to work from scripts directory

$stackName = "battingai"  # Hardcoded stack name
$bucketName = aws cloudformation describe-stacks --stack-name $stackName --query "Stacks[0].Outputs[?OutputKey=='VideoBucket'].OutputValue" --output text

Write-Host "Using S3 bucket: $bucketName"

# Create directories for reference videos
New-Item -Path "..\reference_videos" -ItemType Directory -Force

# Copy reference videos (adjust paths as needed)
Write-Host "Copying reference videos..."
Copy-Item -Path "C:\Users\jcewa\Documents\battingai\bryce.mp4" -Destination "..\reference_videos\bryce_harper.mp4" -ErrorAction SilentlyContinue
Copy-Item -Path "C:\Users\jcewa\Documents\battingai\bryce.mp4" -Destination "..\reference_videos\brandon_lowe.mp4" -ErrorAction SilentlyContinue

# Upload reference videos using the updated script that uses the same functionality as user videos
Write-Host "Uploading reference videos..."
python ..\scripts\upload_reference_videos.py --bucket $bucketName --video ..\reference_videos\bryce_harper.mp4 --player-id bryce_harper --player-name "Bryce Harper"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upload Bryce Harper reference video"
    exit 1
}

python ..\scripts\upload_reference_videos.py --bucket $bucketName --video ..\reference_videos\brandon_lowe.mp4 --player-id brandon_lowe --player-name "Brandon Lowe"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upload Brandon Lowe reference video"
    exit 1
}

Write-Host "Reference data setup complete!"