# test_opencv.py
import cv2
import numpy as np
import os
import sys
import tempfile

def test_opencv_installation():
    """Test if OpenCV is installed correctly"""
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV path: {cv2.__file__}")
    
    # Create a simple image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Blue
    
    # Try to save and read the image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
        temp_path = temp_img.name
    
    try:
        cv2.imwrite(temp_path, img)
        print(f"Successfully wrote test image to {temp_path}")
        
        img_read = cv2.imread(temp_path)
        if img_read is None:
            print("ERROR: Failed to read test image")
        else:
            print(f"Successfully read test image, shape: {img_read.shape}")
    finally:
        os.unlink(temp_path)
        print(f"Cleaned up test image file")

def test_video_processing(video_path=None):
    """Test video processing functionality"""
    if video_path is None:
        # Create a test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_path = temp_video.name
            
        try:
            # Create a simple video with 10 frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30, (640, 480))
            
            for i in range(10):
                # Create a frame with a number
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            print(f"Created test video at {temp_path}")
            
            # Test reading the video
            test_video_reading(temp_path)
        finally:
            os.unlink(temp_path)
            print(f"Cleaned up test video file")
    else:
        # Test reading the provided video
        test_video_reading(video_path)

def test_video_reading(video_path):
    """Test reading a video file"""
    print(f"Testing video reading for {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: frames={frame_count}, fps={fps}, resolution={width}x{height}")
    
    # Read a few frames
    frames_read = 0
    for i in range(min(5, frame_count)):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            print(f"Read frame {i}, shape: {frame.shape}")
        else:
            print(f"Failed to read frame {i}")
    
    cap.release()
    print(f"Successfully read {frames_read} frames")

if __name__ == "__main__":
    print("Testing OpenCV installation...")
    test_opencv_installation()
    
    print("\nTesting video processing...")
    if len(sys.argv) > 1:
        test_video_processing(sys.argv[1])
    else:
        test_video_processing()
