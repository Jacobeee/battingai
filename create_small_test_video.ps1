# PowerShell script to create a smaller test video by extracting a portion of the original

# Path to your original video file
$originalVideoPath = "C:\Users\jcewa\Documents\battingai\user.mp4"
$outputVideoPath = "C:\Users\jcewa\Documents\battingai\user_small.mp4"

# Create a smaller file by copying just a portion of the original
Write-Host "Creating smaller test video..."

try {
    # Read the first 5MB of the original file
    $bytes = [System.IO.File]::ReadAllBytes($originalVideoPath)
    $smallerBytes = $bytes[0..5000000]  # Take first 5MB
    
    # Write to new file
    [System.IO.File]::WriteAllBytes($outputVideoPath, $smallerBytes)
    
    Write-Host "Small test file created at: $outputVideoPath"
    Write-Host "Original size: $((Get-Item $originalVideoPath).Length / 1MB) MB"
    Write-Host "New size: $((Get-Item $outputVideoPath).Length / 1MB) MB"
    
    Write-Host "`nNOTE: This is NOT a valid video file, but can be used for testing the upload process."
}
catch {
    Write-Host "Error creating small test file: $_"
}