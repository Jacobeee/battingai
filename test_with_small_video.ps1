# PowerShell script to test the BattingAI application with a small video file

# Get the API Gateway URL from CloudFormation outputs
$stackName = "battingai"
$apiUrl = aws cloudformation describe-stacks --stack-name $stackName --query "Stacks[0].Outputs[?OutputKey=='BattingAIApi'].OutputValue" --output text
Write-Host "API URL: $apiUrl"

# Path to your small test video file
$videoPath = "C:\Users\jcewa\Documents\battingai\user_small.mp4"

# Step 1: Encode video to base64
Write-Host "Encoding video to base64..."
$videoBytes = [System.IO.File]::ReadAllBytes($videoPath)
$base64Video = [System.Convert]::ToBase64String($videoBytes)
Write-Host "Video encoded. Length: $($base64Video.Length) characters"

# Step 2: Upload video
Write-Host "Uploading video..."
$uploadUrl = $apiUrl + "upload"
$headers = @{
    "Content-Type" = "application/json"
}
$body = "{`"video`": `"$base64Video`"}"

try {
    $uploadResponse = Invoke-RestMethod -Uri $uploadUrl -Method Post -Headers $headers -Body $body
    $analysisId = $uploadResponse.analysis_id
    Write-Host "Upload successful. Analysis ID: $analysisId"
} 
catch {
    Write-Host "Error uploading video: $_"
    exit
}

# Step 3: Start analysis
Write-Host "Starting analysis..."
$analyzeUrl = $apiUrl + "analyze"
$body = "{`"analysis_id`": `"$analysisId`", `"player_id`": `"bryce_harper`"}"

try {
    $analyzeResponse = Invoke-RestMethod -Uri $analyzeUrl -Method Post -Headers $headers -Body $body
    Write-Host "Analysis started. Status: $($analyzeResponse.status)"
}
catch {
    Write-Host "Error starting analysis: $_"
    exit
}

# Step 4: Wait for processing
Write-Host "Waiting for processing to complete..."
$processingComplete = $false
$resultsUrl = $apiUrl + "results/$analysisId"
$maxAttempts = 20
$attempt = 0

while (-not $processingComplete -and $attempt -lt $maxAttempts) {
    $attempt++
    Write-Host "Checking status (attempt $attempt of $maxAttempts)..."
    
    try {
        Start-Sleep -Seconds 15
        $statusResponse = Invoke-RestMethod -Uri $resultsUrl -Method Get
        
        if ($statusResponse.status -eq "feedback_generated") {
            $processingComplete = $true
            Write-Host "Processing complete!"
        }
        else {
            Write-Host "Current status: $($statusResponse.status). Waiting..."
        }
    }
    catch {
        Write-Host "Error checking status: $_"
        Start-Sleep -Seconds 5
    }
}

# Step 5: Get results
if ($processingComplete) {
    Write-Host "Getting final results..."
    try {
        $results = Invoke-RestMethod -Uri $resultsUrl -Method Get
        
        # Save results to a file
        $results | ConvertTo-Json -Depth 10 | Out-File -FilePath "analysis_results_$analysisId.json"
        
        Write-Host "Results saved to analysis_results_$analysisId.json"
        Write-Host "Summary: $($results.results.feedback.summary)"
        
        # Display issues found
        if ($results.results.feedback.issues.Count -gt 0) {
            Write-Host "`nIssues found:"
            foreach ($issue in $results.results.feedback.issues) {
                Write-Host "- $($issue.type): $($issue.description)"
            }
        }
    }
    catch {
        Write-Host "Error getting results: $_"
    }
}
else {
    Write-Host "Processing did not complete within the expected time."
}

Write-Host "`nTest complete!"