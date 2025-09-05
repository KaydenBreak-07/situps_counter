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
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            yield "data: {\"error\": \"Could not open video\"}\n\n"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        while processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Pose detection
            keypoints, results = pose_analyzer.get_keypoints(frame)

            # Sit-up analysis
            torso_angle = analyzer.analyze_situp(keypoints, frame.shape)
            feedback = analyzer.last_feedback  # comes from analyzer state
            counts = analyzer.get_counts()

            # Draw landmarks
            frame = pose_analyzer.draw_landmarks(frame, results)

            # Overlay text
            cv2.putText(frame, f"Correct: {counts['correct']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Incorrect: {counts['incorrect']}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, feedback, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if torso_angle is not None:
                cv2.putText(frame, f"Angle: {torso_angle:.1f}°", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Debug info
            debug_info = f"Keypoints: {len(keypoints)}" if keypoints else "No pose detected"
            cv2.putText(frame, debug_info, (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Build data packet
            debug_data = analyzer.get_debug_info()
            data = {
                'frame': frame_bytes.hex(),
                'counts': counts,
                'feedback': feedback,
                'angle': round(torso_angle, 1) if torso_angle else None,
                'progress': round((frame_count / total_frames) * 100, 1),
                'debug': debug_info,
                'debug_data': debug_data
            }

            yield f"data: {json.dumps(data)}\n\n"
            frame_count += 1

        cap.release()
        processing = False

        final_counts = analyzer.get_counts()
        yield f"data: {json.dumps({'completed': True, 'final_results': final_counts})}\n\n"

    return Response(generate_frames(), mimetype='text/event-stream')
# Add these routes to your Flask app

@app.route('/get_counts')
def get_current_counts():
    """Get current sit-up counts without processing"""
    global analyzer
    counts = analyzer.get_counts()
    return jsonify(counts)

@app.route('/get_detailed_results')
def get_detailed_results():
    """Get detailed results with recommendations"""
    global analyzer
    results = analyzer.get_detailed_results()
    return jsonify(results)

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    """Reset all counters"""
    global analyzer
    analyzer.reset_counts()
    return jsonify({'message': 'Counts reset successfully'})

@app.route('/export_results')
def export_results():
    """Export results as JSON for external use"""
    global analyzer
    results = {
        'timestamp': datetime.now().isoformat(),
        'counts': analyzer.get_counts(),
        'detailed_results': analyzer.get_detailed_results()
    }
    
    # You could also save to a file here
    filename = f"situp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('results', filename)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return jsonify({
        'message': 'Results exported successfully',
        'filename': filename,
        'results': results
    })

@app.route('/stop')
def stop_processing():
    global processing
    processing = False
    return jsonify({'message': 'Processing stopped'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)