import cv2
import argparse
import json
from tqdm import tqdm
from pose_utils import PoseAnalyzer, SitUpAnalyzer

def process_video(input_path, output_path=None):
    """Process video and count sit-ups"""
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize analyzers
    pose_analyzer = PoseAnalyzer()
    situp_analyzer = SitUpAnalyzer()
    
    # Process video frame by frame
    frame_count = 0
    feedback_text = "Starting analysis..."
    
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze pose
        landmarks = pose_analyzer.analyze_frame(frame)
        keypoints = pose_analyzer.get_landmark_coordinates(landmarks, frame.shape)
        
        # Analyze sit-up
        feedback, rep_completed = situp_analyzer.analyze_situp(keypoints, frame.shape)
        feedback_text = feedback
        
        # Draw landmarks and feedback
        if landmarks:
            frame = pose_analyzer.draw_landmarks(frame, landmarks)
        
        # Display counts and feedback
        counts = situp_analyzer.get_counts()
        cv2.putText(frame, f"Correct: {counts['correct']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Incorrect: {counts['incorrect']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, feedback_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to output video if specified
        if output_path:
            out.write(frame)
            
        frame_count += 1
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    # Return results
    results = situp_analyzer.get_counts()
    results['total_frames'] = frame_count
    results['video_duration'] = frame_count / fps if fps > 0 else 0
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Sit-Up Counter using Pose Estimation')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', help='Output video file path (optional)')
    parser.add_argument('--report', help='Output report JSON file path (optional)')
    
    args = parser.parse_args()
    
    print(f"Processing video: {args.input}")
    results = process_video(args.input, args.output)
    
    if results:
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Total sit-ups: {results['total']}")
        print(f"Correct sit-ups: {results['correct']}")
        print(f"Incorrect sit-ups: {results['incorrect']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Video duration: {results['video_duration']:.2f} seconds")
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Report saved to: {args.report}")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    main()