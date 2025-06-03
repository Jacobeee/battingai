# Reference Video Processing

This document explains how reference videos are processed in the BattingAI system.

## Overview

Reference videos are professional baseball player swings that are used as a comparison point for user-uploaded videos. To ensure accurate comparisons, reference videos are now processed using the same functionality as user videos.

## Changes Made

The following changes have been implemented:

1. Updated `scripts/upload_reference_videos.py` to:
   - Import frame extraction and processing functions from `battingai-opencv-lambda/app.py`
   - Use the same frame extraction algorithm as user videos
   - Use the same S3 storage format with progressive loading support

2. Updated `battingai-opencv-lambda/app.py` to:
   - Enhance the `get_reference_frames` function to handle both old and new reference frame formats
   - Support metadata-based frame retrieval for better performance

3. Added error handling and validation to ensure consistent processing

## How to Use

### Uploading Reference Videos

Use the `upload_reference_videos.py` script to upload reference videos:

```bash
python scripts/upload_reference_videos.py --bucket YOUR_BUCKET_NAME --video path/to/video.mp4 --player-id player_name --player-name "Player Name"
```

Parameters:
- `--bucket`: S3 bucket name
- `--video`: Path to the video file
- `--player-id`: Unique identifier for the player (e.g., bryce_harper)
- `--player-name`: Display name for the player (e.g., "Bryce Harper")
- `--profile`: (Optional) AWS profile name
- `--region`: (Optional) AWS region (default: us-east-1)

### Setting Up Multiple Reference Videos

Use the setup scripts to upload multiple reference videos:

For Linux/macOS:
```bash
./scripts/setup_reference_data.sh
```

For Windows:
```powershell
.\scripts\setup_reference_data.ps1
```

### Testing Reference Video Processing

Use the test script to verify that reference videos are processed correctly:

```bash
python test_reference_upload.py --bucket YOUR_BUCKET_NAME --video path/to/video.mp4 --player-id test_player --player-name "Test Player"
```

## Technical Details

### Frame Extraction Process

Reference videos now use the same frame extraction process as user videos:

1. Video is loaded and frames are extracted
2. Frame quality validation is performed
3. Motion analysis is used to identify key frames
4. Swing phases are detected (setup, load, swing, contact, follow-through)
5. Frames are saved with multiple resolution variants for progressive loading

### S3 Storage Format

Reference frames are now stored in the following format:

- Video: `reference/{player_id}.mp4`
- Frames: `reference/{player_id}/frames/frame_{index}_{size}.jpg`
  - Where `size` is one of: `thumbnail`, `medium`, or `full`
- Metadata: `reference/{player_id}/metadata.json`

### Metadata Structure

The metadata JSON file contains:
- `player_id`: Unique identifier for the player
- `player_name`: Display name for the player
- `video_key`: S3 key for the original video
- `frame_paths`: Array of S3 keys for the frames
- `frame_urls`: Array of pre-signed URLs for the frames
- `frame_count`: Number of frames
- `created_at`: Timestamp when the reference was created

## Troubleshooting

If you encounter issues:

1. Check the logs for error messages
2. Verify that the video file is valid and accessible
3. Ensure AWS credentials have the necessary permissions
4. Try running the test script to isolate the issue