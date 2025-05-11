import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modular components
from models.yolo_detector import YOLODetector
from models.resnet_crowd_counter import CrowdCounter
from models.groq_analyzer import GroqAnalyzer
from models.platform_analyzer import PlatformAnalyzer
from models.metro_analyzer import MetroAnalyzer
from models.utils import allowed_file

app = Flask(__name__)

# Groq API configuration
GROQ_API_KEY = "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"

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("uploads/platforms", exist_ok=True)  # New directory for platform images
os.makedirs("models", exist_ok=True)  # Directory for model weights

# Initialize models
logger.info("Initializing YOLO detector...")
yolo_detector = YOLODetector("Yolo-Weights/yolov8l.pt")
logger.info("YOLO detector initialized successfully!")

logger.info("Initializing other models...")
crowd_counter = CrowdCounter()
groq_analyzer = GroqAnalyzer(GROQ_API_KEY)
platform_analyzer = PlatformAnalyzer(yolo_detector)
metro_analyzer = MetroAnalyzer(yolo_detector, groq_analyzer)
logger.info("All models initialized successfully!")

# Set upload file size limit
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Add debug logging
    logger.debug("Request files keys: %s", list(request.files.keys()))
    logger.debug("Request form keys: %s", list(request.form.keys()))
    
    if 'files[]' not in request.files:
        # Check if files are coming in with a different name
        if len(request.files) > 0:
            # Use whatever files are available
            files = []
            for key in request.files:
                if key.endswith('[]') or 'file' in key.lower():
                    files.extend(request.files.getlist(key))
            if not files:
                # If still no valid files, use all files
                for key in request.files:
                    files.extend(request.files.getlist(key))
        else:
            return jsonify({'error': 'No files uploaded. Please select at least one image file.'})
    else:
        files = request.files.getlist('files[]')
    
    # Verify we have valid files
    if not files or len(files) == 0 or all(file.filename == '' for file in files):
        return jsonify({'error': 'No valid files selected'})
    
    # Get bogie IDs if provided
    bogie_ids = None
    if 'bogie_ids[]' in request.form:
        try:
            bogie_ids = [int(bid) - 1 for bid in request.form.getlist('bogie_ids[]')]  # Convert to 0-based index
        except ValueError:
            logger.error("Error converting bogie IDs to integers")
            # Fall back to default IDs
            bogie_ids = list(range(len(files)))
    
    # Save uploaded files
    image_paths = []
    for file in files:
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            image_paths.append(filepath)
            logger.debug("Saved file: %s", filepath)
    
    if not image_paths:
        return jsonify({'error': 'No valid image files uploaded'})
    
    # Process platform image if provided
    platform_data = None
    if 'platform_image' in request.files:
        platform_file = request.files['platform_image']
        if platform_file and platform_file.filename != '' and allowed_file(platform_file.filename):
            platform_filename = secure_filename(platform_file.filename)
            platform_filepath = os.path.join('uploads/platforms', platform_filename)
            platform_file.save(platform_filepath)
            logger.debug("Saved platform file: %s", platform_filepath)
            platform_data = platform_analyzer.analyze_platform_image(platform_filepath)
    
    # Process metro images with bogie IDs
    logger.info("Processing %d metro images...", len(image_paths))
    results = metro_analyzer.process_metro_images(image_paths, bogie_ids)
    logger.info("Metro images processed successfully!")
    
    # Add platform data to results
    if platform_data:
        results['platform_data'] = platform_data
    
    return jsonify(results)

@app.route('/analyze_platform', methods=['POST'])
def analyze_platform():
    if 'platform_image' not in request.files:
        return jsonify({'error': 'No platform image uploaded'})
    
    platform_file = request.files['platform_image']
    
    if platform_file.filename == '' or not allowed_file(platform_file.filename):
        return jsonify({'error': 'No valid platform image selected'})
    
    # Save uploaded file
    platform_path = os.path.join('uploads/platforms', 'platform.jpg')
    platform_file.save(platform_path)
    logger.debug("Saved platform file: %s", platform_path)
    
    # Analyze platform image
    logger.info("Analyzing platform image...")
    platform_data = platform_analyzer.analyze_platform_image(platform_path)
    logger.info("Platform image analyzed successfully!")
    
    if not platform_data:
        return jsonify({'error': 'Failed to analyze platform image'})
    
    return jsonify(platform_data)

if __name__ == '__main__':
    app.run(debug=True)