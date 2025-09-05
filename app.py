from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
from werkzeug.utils import secure_filename
from pose_utils import PoseAnalyzer, SitUpAnalyzer
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for video processing
current_video_path = None
processing = False
analyzer = SitUpAnalyzer()
pose_analyzer = PoseAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video_path, processing, analyzer
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if file:
        # Reset analyzer for new video
        analyzer = SitUpAnalyzer()
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{timestamp}_{filename}"
        current_video_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(current_video_path)
        
        return jsonify({'message': 'Video uploaded successfully', 'filename': save_filename})

@app.route('/process')
def process_video():
    global current_video_path, processing, analyzer, pose_analyzer
    
    if not current_video_path or not os.path.exists(current_video_path):
        return jsonify({'error': 'No video available for processing'}), 400
    
    def generate_frames():
        global processing, analyzer, pose_analyzer
        
        processing = True
        
        # Initialize video capture
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            yield "data: {\"error\": \"Could not open video\"}\n\n"
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        
        while processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze pose
            landmarks = pose_analyzer.analyze_frame(frame)
            keypoints = pose_analyzer.get_landmark_coordinates(landmarks, frame.shape)
            
            # Analyze sit-up
            feedback, rep_completed, torso_angle = analyzer.analyze_situp(keypoints, frame.shape)
            
            # Draw landmarks and feedback
            if landmarks:
                frame = pose_analyzer.draw_landmarks(frame, landmarks)
            
            # Display counts and feedback
            counts = analyzer.get_counts()
            cv2.putText(frame, f"Correct: {counts['correct']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Incorrect: {counts['incorrect']}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, feedback, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {torso_angle:.1f}°", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add debug info
            debug_info = ""
            if not landmarks:
                debug_info = "No pose detected"
            elif len(keypoints) < 4:
                debug_info = f"Not enough keypoints: {len(keypoints)}/4"
            else:
                debug_info = f"Keypoints: {len(keypoints)}"
            
            cv2.putText(frame, debug_info, (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Get debug info
            debug_data = analyzer.get_debug_info()
            
            # Send frame and data
            data = {
                'frame': frame_bytes.hex(),
                'counts': counts,
                'feedback': feedback,
                'angle': round(torso_angle, 1),
                'progress': round((frame_count / total_frames) * 100, 1),
                'debug': debug_info,
                'debug_data': debug_data
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            frame_count += 1
        
        cap.release()
        processing = False
        
        # Send final results
        final_counts = analyzer.get_counts()
        yield f"data: {json.dumps({'completed': True, 'final_results': final_counts})}\n\n"
    
    return Response(generate_frames(), mimetype='text/event-stream')

@app.route('/stop')
def stop_processing():
    global processing
    processing = False
    return jsonify({'message': 'Processing stopped'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)