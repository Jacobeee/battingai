import cv2
import numpy as np
import os
import json
import argparse
from datetime import datetime

def extract_frames(video_path, num_frames=10):
    """Extract key frames from a video file"""
    print(f"Extracting frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")
    
    # Calculate frame indices to extract (evenly distributed)
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            # Save frame for visualization
            cv2.imwrite(f"frame_{len(frames)}.jpg", frame)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames

def compare_frames(user_frames, reference_frames):
    """Compare user frames with reference frames using histogram similarity and more detailed analysis"""
    print("Comparing frames...")
    results = []
    
    # Ensure we have the same number of frames to compare
    min_frames = min(len(user_frames), len(reference_frames))
    
    for i in range(min_frames):
        user_frame = user_frames[i]
        ref_frame = reference_frames[i]
        
        # Calculate histogram similarity
        user_hist = cv2.calcHist([user_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        ref_hist = cv2.calcHist([ref_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(user_hist, user_hist)
        cv2.normalize(ref_hist, ref_hist)
        
        # Compare histograms
        similarity = cv2.compareHist(user_hist, ref_hist, cv2.HISTCMP_CORREL)
        
        # Basic image analysis for stance detection
        user_height, user_width = user_frame.shape[:2]
        ref_height, ref_width = ref_frame.shape[:2]
        
        # Calculate lower half of image (where feet would be)
        user_lower_half = user_frame[user_height//2:, :]
        ref_lower_half = ref_frame[ref_height//2:, :]
        
        # Convert to grayscale for edge detection
        user_gray = cv2.cvtColor(user_lower_half, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_lower_half, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        user_edges = cv2.Canny(user_gray, 100, 200)
        ref_edges = cv2.Canny(ref_gray, 100, 200)
        
        # Count edge pixels as a simple measure of stance width
        user_edge_count = np.count_nonzero(user_edges)
        ref_edge_count = np.count_nonzero(ref_edges)
        
        # Calculate edge ratio
        edge_ratio = user_edge_count / ref_edge_count if ref_edge_count > 0 else 1.0
        
        # Save edge detection images for visualization
        if i < 3:  # Only save for stance frames
            cv2.imwrite(f"user_edges_{i+1}.jpg", user_edges)
            cv2.imwrite(f"ref_edges_{i+1}.jpg", ref_edges)
        
        results.append({
            'frame_index': i,
            'similarity_score': float(similarity),
            'edge_ratio': float(edge_ratio),
            'issues': []
        })
        
        # Add detailed issues based on analysis
        if similarity < 0.8:
            if i < 3:  # Early frames - setup and stance
                # Determine specific stance issues
                if edge_ratio < 0.8:
                    results[i]['issues'].append({
                        'type': 'stance_narrow',
                        'description': 'Your stance is too narrow compared to the reference'
                    })
                elif edge_ratio > 1.2:
                    results[i]['issues'].append({
                        'type': 'stance_wide',
                        'description': 'Your stance is too wide compared to the reference'
                    })
                else:
                    # Check for weight distribution issues using color distribution
                    user_left = user_lower_half[:, :user_width//2]
                    user_right = user_lower_half[:, user_width//2:]
                    
                    left_intensity = np.mean(user_left)
                    right_intensity = np.mean(user_right)
                    
                    if abs(left_intensity - right_intensity) > 10:
                        results[i]['issues'].append({
                            'type': 'stance_weight',
                            'description': 'Your weight distribution appears uneven'
                        })
                    else:
                        results[i]['issues'].append({
                            'type': 'stance_posture',
                            'description': 'Your stance posture needs adjustment'
                        })
            elif i < 6:  # Middle frames - swing initiation
                results[i]['issues'].append({
                    'type': 'hip_rotation',
                    'description': 'Your hip rotation is delayed compared to the reference'
                })
            else:  # Late frames - follow through
                results[i]['issues'].append({
                    'type': 'follow_through',
                    'description': 'Your follow-through is incomplete compared to the reference'
                })
    
    return results

def generate_feedback(comparison_results):
    """Generate detailed feedback based on comparison results"""
    print("Generating feedback...")
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
            
            # Define specific causes and corrections based on issue type
            causes = []
            corrections = []
            
            if issue_type == 'stance_narrow':
                causes = [
                    "Feet positioned too close together",
                    "Attempting to maintain too much balance",
                    "Improper understanding of batting stance fundamentals"
                ]
                corrections = [
                    "Widen your stance to approximately shoulder width",
                    "Position feet to create a stable base",
                    "Practice proper stance width with a batting tee"
                ]
            elif issue_type == 'stance_wide':
                causes = [
                    "Feet positioned too far apart",
                    "Overcompensating for balance issues",
                    "Incorrect attempt to generate power"
                ]
                corrections = [
                    "Narrow your stance to approximately shoulder width",
                    "Focus on a balanced athletic position",
                    "Practice proper stance width with a batting tee"
                ]
            elif issue_type == 'stance_weight':
                causes = [
                    "Leaning too much on front or back foot",
                    "Improper weight distribution",
                    "Poor balance in setup position"
                ]
                corrections = [
                    "Distribute weight evenly between both feet",
                    "Maintain a balanced athletic position",
                    "Practice with weight shift drills"
                ]
            elif issue_type == 'stance_posture':
                causes = [
                    "Improper spine angle",
                    "Shoulders not level",
                    "Head position not optimal"
                ]
                corrections = [
                    "Maintain a slight bend in the knees",
                    "Keep shoulders level and relaxed",
                    "Position head to see both eyes in mirror"
                ]
            elif issue_type == 'hip_rotation':
                causes = [
                    "Upper body moving before the hips",
                    "Insufficient hip rotation",
                    "Poor sequencing of body movements"
                ]
                corrections = [
                    "Initiate swing with lower body",
                    "Focus on rotating hips toward pitcher",
                    "Practice hip rotation drills"
                ]
            elif issue_type == 'follow_through':
                causes = [
                    "Stopping swing too early",
                    "Not extending arms fully",
                    "Poor weight transfer through swing"
                ]
                corrections = [
                    "Complete full swing motion",
                    "Extend arms through contact",
                    "Transfer weight to front foot during follow-through"
                ]
            else:
                # Default causes and corrections
                causes = [
                    "Improper technique",
                    "Lack of practice with proper form"
                ]
                corrections = [
                    "Study proper technique",
                    "Practice with focused drills"
                ]
            
            # Add feedback for this issue type
            issue_feedback = {
                'type': issue_type,
                'description': issue['description'],
                'causes': causes,
                'corrections': corrections,
                'resources': [
                    {
                        'type': 'youtube',
                        'title': f"Perfect Baseball {issue_type.title()} Tutorial",
                        'url': f"https://www.youtube.com/watch?v=example{len(feedback['issues']) + 1}"
                    }
                ]
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze baseball batting videos')
    parser.add_argument('--user-video', type=str, required=True, help='Path to user video file')
    parser.add_argument('--reference-video', type=str, required=True, help='Path to reference video file')
    parser.add_argument('--output', type=str, default='analysis_results.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    user_video = args.user_video
    reference_video = args.reference_video
    output_file = args.output
    
    # Check if video files exist
    if not os.path.exists(user_video):
        print(f"Error: User video file {user_video} not found")
        return
    
    if not os.path.exists(reference_video):
        print(f"Error: Reference video file {reference_video} not found")
        return
    
    # Extract frames
    user_frames = extract_frames(user_video)
    reference_frames = extract_frames(reference_video)
    
    if not user_frames or not reference_frames:
        print("Error: Failed to extract frames from videos")
        return
    
    # Compare frames
    comparison_results = compare_frames(user_frames, reference_frames)
    
    # Generate feedback
    feedback = generate_feedback(comparison_results)
    
    # Save results to a file
    results = {
        'timestamp': datetime.now().isoformat(),
        'comparison_results': comparison_results,
        'feedback': feedback
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    print(f"Summary: {feedback['summary']}")
    
    # Print detailed issues
    if feedback['issues']:
        print("\nIssues found:")
        for issue in feedback['issues']:
            print(f"\n- {issue['type']}: {issue['description']}")
            print("  Causes:")
            for cause in issue['causes']:
                print(f"    * {cause}")
            print("  Corrections:")
            for correction in issue['corrections']:
                print(f"    * {correction}")
            print("  Resources:")
            for resource in issue['resources']:
                print(f"    * {resource['title']}: {resource['url']}")

if __name__ == "__main__":
    main()
