# Test script for BattingAI API using direct S3 upload

# Get the S3 bucket name from CloudFormation outputs
$stackName = "battingai"
try {
    $bucketName = aws cloudformation describe-stacks --stack-name $stackName --query "Stacks[0].Outputs[?OutputKey=='VideoBucket'].OutputValue" --output text
    if ([string]::IsNullOrEmpty($bucketName)) {
        throw "Bucket name not found"
    }
} catch {
    # Fallback to the hardcoded bucket name if CloudFormation query fails
    $bucketName = "battingai-videobucket-ayk9m1uehbg2"
}

$apiUrl = "https://15zjlknmve.execute-api.us-east-1.amazonaws.com/Prod/"

Write-Host "Using S3 bucket: $bucketName"
Write-Host "Using API URL: $apiUrl"

# Generate a unique analysis ID
$analysisId = [guid]::NewGuid().ToString()
Write-Host "Generated Analysis ID: $analysisId"

# Create a small test video file
$testVideoPath = "test_video.mp4"
if (-not (Test-Path $testVideoPath)) {
    # Create a minimal MP4 file
    [byte[]]$bytes = 0..255 | Get-Random -Count 1024
    [System.IO.File]::WriteAllBytes($testVideoPath, $bytes)
    Write-Host "Created test video file: $testVideoPath"
}

# Step 1: Upload video directly to S3
$videoKey = "uploads/test_$(Get-Date -Format 'yyyyMMddHHmmss').mp4"
Write-Host "Uploading video to S3: $videoKey"
aws s3 cp $testVideoPath s3://$bucketName/$videoKey

# Step 2: Create metadata in S3
$metadata = @{
    analysis_id = $analysisId
    video_key = $videoKey
    timestamp = (Get-Date -Format 'yyyyMMddHHmmss')
    status = "uploaded"
    # Add mock frame paths for testing
    frame_paths = @(
        "analyses/$analysisId/frames/frame_0.jpg",
        "analyses/$analysisId/frames/frame_1.jpg",
        "analyses/$analysisId/frames/frame_2.jpg",
        "analyses/$analysisId/frames/frame_3.jpg",
        "analyses/$analysisId/frames/frame_4.jpg",
        "analyses/$analysisId/frames/frame_5.jpg",
        "analyses/$analysisId/frames/frame_6.jpg",
        "analyses/$analysisId/frames/frame_7.jpg",
        "analyses/$analysisId/frames/frame_8.jpg",
        "analyses/$analysisId/frames/frame_9.jpg"
    )
} | ConvertTo-Json

$metadataPath = "temp_metadata.json"
$metadata | Out-File -FilePath $metadataPath
Write-Host "Uploading metadata to S3"
aws s3 cp $metadataPath s3://$bucketName/analyses/$analysisId/metadata.json
Remove-Item -Path $metadataPath

# Step 3: Call the analyze endpoint
Write-Host "Calling analyze endpoint for analysis ID: $analysisId"
$analyzeUrl = $apiUrl + "analyze"
$analyzeBody = @{
    analysis_id = $analysisId
    player_id = "bryce_harper"
} | ConvertTo-Json

try {
    $analyzeResponse = Invoke-RestMethod -Uri $analyzeUrl -Method Post -ContentType "application/json" -Body $analyzeBody
    Write-Host "Analyze Response:"
    $analyzeResponse | ConvertTo-Json
}
catch {
    Write-Host "Error calling analyze endpoint: $_"
    Write-Host "Continuing anyway to check results..."
}

# Step 4: Wait for processing to complete
Write-Host "Waiting for processing to complete..."
$maxAttempts = 5
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $attempt++
    Write-Host "Checking status (attempt $attempt of $maxAttempts)..."
    Start-Sleep -Seconds 5
    
    # Check the results endpoint
    $resultsUrl = $apiUrl + "results/$analysisId"
    
    try {
        $resultsResponse = Invoke-RestMethod -Uri $resultsUrl -Method Get
        Write-Host "Current status: $($resultsResponse.status)"
        
        if ($resultsResponse.status -eq "feedback_generated") {
            Write-Host "Processing complete!"
            Write-Host "Results:"
            $resultsResponse | ConvertTo-Json -Depth 4
            break
        }
    }
    catch {
        Write-Host "Error checking results: $_"
    }
}

# Step 5: Check S3 for any results
Write-Host "Checking S3 for analysis artifacts..."
aws s3 ls s3://$bucketName/analyses/$analysisId/ --recursive

Write-Host "Test complete!"