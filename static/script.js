document.addEventListener('DOMContentLoaded', function() {
    const videoUpload = document.getElementById('videoUpload');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const stopBtn = document.getElementById('stopBtn');
    const videoCanvas = document.getElementById('videoCanvas');
    const ctx = videoCanvas.getContext('2d');
    
    const correctCount = document.getElementById('correctCount');
    const incorrectCount = document.getElementById('incorrectCount');
    const totalCount = document.getElementById('totalCount');
    const accuracy = document.getElementById('accuracy');
    const angle = document.getElementById('angle');
    const feedback = document.getElementById('feedback');
    const debugInfo = document.getElementById('debugInfo');
    const stateInfo = document.getElementById('stateInfo');
    const missingPoints = document.getElementById('missingPoints');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    const finalResults = document.getElementById('finalResults');
    const finalCorrect = document.getElementById('finalCorrect');
    const finalIncorrect = document.getElementById('finalIncorrect');
    const finalTotal = document.getElementById('finalTotal');
    const finalAccuracy = document.getElementById('finalAccuracy');
    
    let eventSource = null;
    
    // Handle video upload
    uploadBtn.addEventListener('click', function() {
        if (!videoUpload.files.length) {
            alert('Please select a video file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('video', videoUpload.files[0]);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                alert('Video uploaded successfully!');
                processBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error uploading video');
        });
    });
    
    // Start processing
    processBtn.addEventListener('click', function() {
        processBtn.disabled = true;
        stopBtn.disabled = false;
        finalResults.style.display = 'none';
        
        // Reset stats
        correctCount.textContent = '0';
        incorrectCount.textContent = '0';
        totalCount.textContent = '0';
        accuracy.textContent = '0%';
        angle.textContent = '0°';
        feedback.textContent = 'Starting analysis...';
        
        // Set up Server-Sent Events connection
        eventSource = new EventSource('/process');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                alert('Error: ' + data.error);
                stopProcessing();
                return;
            }
            
            if (data.completed) {
                // Show final results
                showFinalResults(data.final_results);
                stopProcessing();
                return;
            }
            
            // Update frame
            if (data.frame) {
                const frameBytes = new Uint8Array(data.frame.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
                const blob = new Blob([frameBytes], {type: 'image/jpeg'});
                const url = URL.createObjectURL(blob);
                
                const img = new Image();
                img.onload = function() {
                    videoCanvas.width = img.width;
                    videoCanvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    URL.revokeObjectURL(url);
                };
                img.src = url;
            }
            
            // Update stats
            if (data.counts) {
                correctCount.textContent = data.counts.correct;
                incorrectCount.textContent = data.counts.incorrect;
                totalCount.textContent = data.counts.total;
                accuracy.textContent = data.counts.accuracy + '%';
            }
            
            if (data.angle) {
                angle.textContent = data.angle + '°';
            }
            
            if (data.feedback) {
                feedback.textContent = data.feedback;
            }
            
            if (data.debug) {
                debugInfo.textContent = data.debug;
            }
            
            if (data.debug_data) {
                stateInfo.textContent = data.debug_data.state;
                missingPoints.textContent = data.debug_data.missing_keypoints.join(', ') || 'None';
            }
            
            if (data.progress) {
                progressFill.style.width = data.progress + '%';
                progressText.textContent = data.progress + '%';
            }
        };
        
        eventSource.onerror = function() {
            console.error('EventSource failed');
            stopProcessing();
        };
    });
    
    // Stop processing
    stopBtn.addEventListener('click', function() {
        stopProcessing();
        fetch('/stop');
    });
    
    function stopProcessing() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        
        processBtn.disabled = false;
        stopBtn.disabled = true;
    }
    
    function showFinalResults(results) {
        finalCorrect.textContent = results.correct;
        finalIncorrect.textContent = results.incorrect;
        finalTotal.textContent = results.total;
        finalAccuracy.textContent = results.accuracy + '%';
        finalResults.style.display = 'block';
    }
});