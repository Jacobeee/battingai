# Test Lambda functions directly using base64-encoded payloads

# Test ProcessVideoFunction
Write-Host "Testing ProcessVideoFunction..."
$processPayload = '{"analysis_id":"test-123","video_key":"uploads/test.mp4"}'
$processPayloadBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($processPayload))
aws lambda invoke --function-name battingai-ProcessVideoFunction-4Prwb1gpOv0Q --payload $processPayloadBase64 output_process.json

# Test CompareVideosFunction
Write-Host "Testing CompareVideosFunction..."
$comparePayload = '{"analysis_id":"test-123","player_id":"bryce_harper"}'
$comparePayloadBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($comparePayload))
aws lambda invoke --function-name battingai-CompareVideosFunction-K8UfVH29w0X7 --payload $comparePayloadBase64 output_compare.json

# Test GenerateFeedbackFunction
Write-Host "Testing GenerateFeedbackFunction..."
$feedbackPayload = '{"analysis_id":"test-123"}'
$feedbackPayloadBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($feedbackPayload))
aws lambda invoke --function-name battingai-GenerateFeedbackFunction-ABa99UJ9auZF --payload $feedbackPayloadBase64 output_feedback.json

# Check the outputs
Write-Host "Process Video Output:"
if (Test-Path output_process.json) { Get-Content output_process.json }
else { Write-Host "No output file found" }

Write-Host "Compare Videos Output:"
if (Test-Path output_compare.json) { Get-Content output_compare.json }
else { Write-Host "No output file found" }

Write-Host "Generate Feedback Output:"
if (Test-Path output_feedback.json) { Get-Content output_feedback.json }
else { Write-Host "No output file found" }
