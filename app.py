import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename  # Add this line
import requests
import json
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models
from scipy.ndimage import gaussian_filter

app = Flask(__name__)

# Load YOLOv8 model (we'll keep this for object detection tasks)
model = YOLO("Yolo-Weights/yolov8l.pt")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("uploads/platforms", exist_ok=True)  # New directory for platform images
os.makedirs("models", exist_ok=True)  # Directory for model weights

# Groq API configuration
GROQ_API_KEY = "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" # You'll need to add your Groq API key here
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Train frequency settings
TRAIN_FREQUENCY = {
    "low": 15,     # 15 minutes between trains during low platform crowd
    "medium": 10,  # 10 minutes between trains during medium platform crowd
    "high": 5      # 5 minutes between trains during high platform crowd
}

# Define crowd density levels for inference
CROWD_LEVELS = {
    "Empty": 10,
    "A few people": 20,
    "Average crowd": 40,
    "Getting busy": 60,
    "Busy": 80,
    "Packed": float('inf')
}

# Define ResNet18 model for crowd counting
class ResNet18CrowdCounter(nn.Module):
    def __init__(self, load_weights=True):
        super(ResNet18CrowdCounter, self).__init__()
        # Load pretrained ResNet18 backbone
        resnet18 = models.resnet18(pretrained=load_weights)
        
        # Modify the first conv layer to accept grayscale input (1 channel)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from pretrained model for all layers except first conv
        if load_weights:
            self.conv1.weight.data = torch.mean(resnet18.conv1.weight.data, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        
        # Use all layers from ResNet except the final FC layer
        self.features = nn.Sequential(
            self.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        )
        
        # Custom regression head for crowd counting
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regression_head(x)
        return x.squeeze()

# Initialize ResNet18 model
crowd_counter_model = None

def load_crowd_counter_model():
    global crowd_counter_model
    # Check if model is already loaded
    if crowd_counter_model is not None:
        return crowd_counter_model
    
    # Initialize model
    crowd_counter_model = ResNet18CrowdCounter(load_weights=True)
    
    # Set model to evaluation mode
    crowd_counter_model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crowd_counter_model = crowd_counter_model.to(device)
    
    return crowd_counter_model

def analyze_with_groq(counts, recommendations, platform_data=None, image_paths=None):
    """Use Groq API to get enhanced analysis of crowd distribution with qualitative assessment"""
    if not GROQ_API_KEY:
        return "Groq API key not configured. Please add your API key to use this feature."
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create a more detailed prompt with specific context
        prompt = f"""Analyze the following metro train crowd data with precision and factual accuracy:
        
Metro Train Analysis:
- Specific train ID: Metro Line {datetime.now().strftime('%Y%m%d-%H%M')}
- Time of analysis: {datetime.now().strftime('%H:%M')}
- Total bogies analyzed: {len(counts)}
- Detailed passenger counts by bogie: {counts}
- Current system recommendations: {recommendations}
        
Important context:
- Each bogie has a maximum capacity of approximately 80-100 passengers
- Standing passengers require ~0.4 square meters of space for comfort
- Seated passengers are often undercounted by 15-20% in algorithmic analysis
- Rush hour threshold is typically 70% of maximum capacity
- Algorithmic detection may miss seated passengers or those partially occluded
"""

        # Add platform data if available
        if platform_data:
            prompt += f"""

Platform crowd data:
- Total people waiting: {platform_data['total_people']}
- Platform crowd density: {platform_data['density_level']}
- Current train frequency: Every {platform_data['train_frequency']} minutes
"""

        # Add count estimation request
        prompt += """

Based on the data provided and your knowledge of metro train crowding patterns:

1. For each bogie, provide an estimated ACTUAL passenger count considering:
   - The algorithmic count provided
   - Likely seated passengers that might be missed
   - Partially occluded passengers in crowded areas
   - Typical undercount percentages in similar scenarios

2. Present your count estimates in this format:
   | Bogie | Algorithmic Count | Estimated Actual Count | Confidence (1-10) |
   |-------|-------------------|------------------------|-------------------|
   | 1     | [count]           | [your estimate]        | [confidence]      |

3. Evaluate if the algorithmic counts seem accurate or if they might be underestimating the actual crowd
4. Assess if there are likely seated passengers or standing passengers that might be missed by object detection
5. Estimate the perceived crowding level from a passenger comfort perspective
6. Suggest if any bogies appear to have hidden or occluded passengers not captured in the counts

Then provide:
1. A detailed analysis of the current crowd distribution
2. Potential bottlenecks or congestion points
3. Optimal redistribution strategy with specific instructions for station staff
4. Prediction of how crowd flow might change in next 15-30 minutes
5. Suggestions for improving passenger comfort based on this data

Format your response in markdown."""
        
        payload = {
            "model": "llama3-70b-8192",  # Use a larger model if available
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Lower temperature for more factual responses
            "max_tokens": 1500,  # Allow for more detailed analysis
            "top_p": 0.9  # Add top_p parameter for better quality
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            # Validate and enhance the analysis
            analysis = validate_groq_analysis(analysis, counts)
            return analysis
        else:
            # Use fallback analysis if API fails
            return generate_fallback_analysis(counts, recommendations, platform_data)
    
    except Exception as e:
        return generate_fallback_analysis(counts, recommendations, platform_data)


def extract_groq_counts(analysis, default_counts):
    """Extract the estimated counts from Groq's analysis"""
    try:
        # Look for the table with estimated counts
        import re
        
        # Pattern to match the table rows with counts
        pattern = r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        
        # Find all matches
        matches = re.findall(pattern, analysis)
        
        if matches:
            # Extract the estimated counts (third column)
            groq_counts = [int(match[2]) for match in matches]
            return groq_counts
        else:
            # If no table found, return the default counts
            return default_counts
    except Exception as e:
        print(f"Error extracting Groq counts: {e}")
        return default_counts

def validate_groq_analysis(analysis, counts):
    """Validate and enhance the Groq analysis to ensure quality"""
    # Check if analysis contains required sections
    required_sections = [
        "crowd distribution", "bottlenecks", "redistribution", 
        "prediction", "improving passenger comfort"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in analysis.lower():
            missing_sections.append(section)
    
    # If sections are missing, add placeholders
    if missing_sections:
        analysis += "\n\n## Additional Analysis\n\n"
        for section in missing_sections:
            analysis += f"### {section.title()}\n\n"
            analysis += f"Based on the data, the {section} analysis indicates...[standard recommendation]\n\n"
    
    # Add factual data summary at the beginning
    summary = "## Summary of Crowd Data\n\n"
    summary += "| Bogie | Passenger Count | Density Level |\n"
    summary += "|-------|-----------------|---------------|\n"
    
    for i, count in enumerate(counts):
        density = "Low" if count < 30 else "Medium" if count < 60 else "High"
        summary += f"| {i+1} | {count} | {density} |\n"
    
    return summary + "\n\n" + analysis


def generate_fallback_analysis(counts, recommendations, platform_data=None):
    """Generate a basic analysis when the API fails"""
    total_passengers = sum(counts)
    avg_per_bogie = total_passengers / len(counts) if counts else 0
    most_crowded = counts.index(max(counts)) + 1 if counts else 0
    least_crowded = counts.index(min(counts)) + 1 if counts else 0
    
    analysis = """## Metro Crowd Analysis (Basic)

### Current Situation
"""
    analysis += f"Total passengers: {total_passengers}\n"
    analysis += f"Average passengers per bogie: {avg_per_bogie:.1f}\n"
    analysis += f"Most crowded bogie: {most_crowded} ({max(counts) if counts else 0} passengers)\n"
    analysis += f"Least crowded bogie: {least_crowded} ({min(counts) if counts else 0} passengers)\n\n"
    
    # Add basic recommendations
    analysis += "### Recommendations\n\n"
    analysis += "* Consider redistributing passengers from the most crowded bogie\n"
    analysis += "* Monitor the situation in real-time for changes\n"
    analysis += "* Ensure all available doors are operational for efficient boarding/alighting\n"
    
    return analysis

def process_metro_images(image_paths, bogie_ids=None):
    """Process metro images and return results using enhanced detection for bogies
    
    Args:
        image_paths: List of paths to bogie images
        bogie_ids: Optional list of bogie identifiers (same length as image_paths)
                  If provided, images with the same ID are considered from the same bogie
    """
    # If no bogie_ids provided, assume each image is a different bogie
    if bogie_ids is None:
        bogie_ids = list(range(len(image_paths)))
    
    # Dictionary to store person features by bogie ID
    bogie_data = {}
    
    # Import the YOLO model if not already imported
    if 'yolo_model' not in globals():
        from ultralytics import YOLO
        global yolo_model
        yolo_model = YOLO("Yolo-Weights/yolov8l.pt")
    
    # First pass: detect people in all images and store their features by bogie
    for i, (img_path, bogie_id) in enumerate(zip(image_paths, bogie_ids)):
        # Initialize bogie data if this is the first image for this bogie
        if bogie_id not in bogie_data:
            bogie_data[bogie_id] = {
                'person_features': [],  # Store features for matching
                'person_boxes': [],     # Store bounding boxes
                'best_image': None,     # Store the image with most detections
                'best_count': 0,        # Store the highest detection count
                'all_images': [],       # Store all processed images
                'all_image_paths': []   # Store all processed image paths
            }
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Create a copy for visualization
        img_display = img.copy()
        
        # YOLO detection with lower confidence threshold for seated passengers
        results = yolo_model(img, conf=0.15)  # Lower threshold from 0.3 to 0.15
        
        # Process YOLO results
        yolo_count = 0
        person_boxes = []
        person_features = []
        
        # Extract all person detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if yolo_model.names[cls] == "person" and conf > 0.15:  # Lower threshold for better detection
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract person region for feature calculation
                    person_roi = img[y1:y2, x1:x2]
                    if person_roi.size == 0:  # Skip if ROI is empty
                        continue
                        
                    # Calculate simple feature vector (color histogram)
                    # This is a basic approach - could be enhanced with more sophisticated features
                    try:
                        # Resize to standard size
                        person_roi = cv2.resize(person_roi, (64, 128))
                        
                        # Calculate color histogram as feature
                        hist_b = cv2.calcHist([person_roi], [0], None, [8], [0, 256])
                        hist_g = cv2.calcHist([person_roi], [1], None, [8], [0, 256])
                        hist_r = cv2.calcHist([person_roi], [2], None, [8], [0, 256])
                        
                        # Normalize and flatten
                        hist_b = cv2.normalize(hist_b, hist_b).flatten()
                        hist_g = cv2.normalize(hist_g, hist_g).flatten()
                        hist_r = cv2.normalize(hist_r, hist_r).flatten()
                        
                        # Combine features
                        feature = np.concatenate([hist_b, hist_g, hist_r])
                        
                        # Store feature and box
                        person_features.append(feature)
                        person_boxes.append((x1, y1, x2, y2))
                        
                        # Draw bounding box
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add confidence score
                        cv2.putText(img_display, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        yolo_count += 1
                    except Exception as e:
                        print(f"Error processing person ROI: {e}")
        
        # Add special detection for seated passengers (similar to original code)
        seated_results = yolo_model(img, conf=0.1)  # Even lower threshold for seated detection
        
        # Process seated detections with different criteria
        for r in seated_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if yolo_model.names[cls] == "person" and conf > 0.1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Check if this is likely a seated person (wider than tall, or in seat position)
                    width, height = x2-x1, y2-y1
                    aspect_ratio = width/height if height > 0 else 0
                    
                    # Seated people often have different aspect ratios
                    if aspect_ratio > 0.7 or y1 > img.shape[0]/2:  # Lower body may be hidden by seats
                        # Check if this is a new detection
                        new_detection = True
                        for ex1, ey1, ex2, ey2 in person_boxes:
                            # Calculate overlap
                            overlap_x1 = max(x1, ex1)
                            overlap_y1 = max(y1, ey1)
                            overlap_x2 = min(x2, ex2)
                            overlap_y2 = min(y2, ey2)
                            
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                box_area = (x2 - x1) * (y2 - y1)
                                
                                if overlap_area / box_area > 0.5:
                                    new_detection = False
                                    break
                        
                        if new_detection:
                            try:
                                # Extract person region for feature calculation
                                person_roi = img[y1:y2, x1:x2]
                                if person_roi.size > 0:  # Skip if ROI is empty
                                    # Resize to standard size
                                    person_roi = cv2.resize(person_roi, (64, 128))
                                    
                                    # Calculate color histogram as feature
                                    hist_b = cv2.calcHist([person_roi], [0], None, [8], [0, 256])
                                    hist_g = cv2.calcHist([person_roi], [1], None, [8], [0, 256])
                                    hist_r = cv2.calcHist([person_roi], [2], None, [8], [0, 256])
                                    
                                    # Normalize and flatten
                                    hist_b = cv2.normalize(hist_b, hist_b).flatten()
                                    hist_g = cv2.normalize(hist_g, hist_g).flatten()
                                    hist_r = cv2.normalize(hist_r, hist_r).flatten()
                                    
                                    # Combine features
                                    feature = np.concatenate([hist_b, hist_g, hist_r])
                                    
                                    # Store feature and box
                                    person_features.append(feature)
                                    person_boxes.append((x1, y1, x2, y2))
                                    
                                    # Draw bounding box (yellow for seated)
                                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                    yolo_count += 1
                            except Exception as e:
                                print(f"Error processing seated person ROI: {e}")
        
        # Apply density-based correction (similar to original code)
        if len(person_boxes) > 0:
            # Calculate average person size
            avg_width = sum(x2-x1 for x1,y1,x2,y2 in person_boxes) / len(person_boxes)
            avg_height = sum(y2-y1 for x1,y1,x2,y2 in person_boxes) / len(person_boxes)
            
            # Calculate color variance (high in crowded scenes with different clothing)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_variance = np.std(hsv[:,:,0])
            
            # Calculate edge density
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # Calculate person area ratio (how much of the image is covered by detected people)
            person_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in person_boxes)
            visible_area = img.shape[0] * img.shape[1] * 0.6  # Assume 60% of image is usable space
            area_ratio = person_area / visible_area
            
            # More aggressive correction for crowded scenes
            if (edge_density > 0.08 or color_variance > 25 or area_ratio > 0.15) and len(person_boxes) < 20:
                # Use tighter spacing factor for crowded bogies (1.2 instead of 1.5)
                estimated_capacity = int(visible_area / (avg_width * avg_height * 1.2))  
                
                # Apply more aggressive correction based on visual indicators
                if edge_density > 0.12 or color_variance > 35 or area_ratio > 0.25:  # Very crowded
                    corrected_count = max(yolo_count, int(estimated_capacity * 0.85))  # Increased from 0.7
                elif edge_density > 0.08 or color_variance > 25 or area_ratio > 0.15:  # Moderately crowded
                    corrected_count = max(yolo_count, int(estimated_capacity * 0.6))   # Increased from 0.4
                
                print(f"Applied enhanced density correction: {yolo_count} → {corrected_count}")
                yolo_count = corrected_count
        
        # Create a heatmap visualization (similar to original code)
        heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        if person_boxes:
            for x1, y1, x2, y2 in person_boxes:
                # Center of the person
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                # Size of the person (for scaling the gaussian)
                person_size = max(x2-x1, y2-y1)
                
                # Create a gaussian blob centered at each person
                sigma = person_size / 3
                
                # Generate a grid of coordinates
                y_grid, x_grid = np.ogrid[-center_y:img.shape[0]-center_y, -center_x:img.shape[1]-center_x]
                # Create a gaussian mask
                mask = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2*sigma*sigma))
                heatmap += mask
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert heatmap to color image
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_display, 0.7, heatmap_color, 0.3, 0)
        
        # Determine crowd level based on YOLO count
        crowd_level = "Empty"
        for level, threshold in CROWD_LEVELS.items():
            if yolo_count < threshold:
                crowd_level = level
                break
        
        # Add count text to image
        cv2.putText(overlay, f"Bogie {bogie_id+1}: {yolo_count} people", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(overlay, f"Status: {crowd_level}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save the processed image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("static/results", f"bogie_{bogie_id+1}_view{i+1}_{timestamp}.jpg")
        cv2.imwrite(output_path, overlay)
        
        # Store data for this bogie
        bogie_data[bogie_id]['person_features'].extend(person_features)
        bogie_data[bogie_id]['person_boxes'].extend(person_boxes)
        bogie_data[bogie_id]['all_images'].append(overlay)
        bogie_data[bogie_id]['all_image_paths'].append(output_path.replace("static/", ""))
        
        # Update best image if this has more detections
        if yolo_count > bogie_data[bogie_id]['best_count']:
            bogie_data[bogie_id]['best_count'] = yolo_count
            bogie_data[bogie_id]['best_image'] = overlay
    
    # Second pass: match people across images of the same bogie to avoid double counting
    counts = []
    processed_images = []
    processed_image_paths = []
    
    for bogie_id in sorted(bogie_data.keys()):
        # If we have multiple images for this bogie, perform matching
        if len(bogie_data[bogie_id]['all_images']) > 1:
            # Get all features from this bogie
            all_features = np.array(bogie_data[bogie_id]['person_features'])
            
            # Skip if no features
            if len(all_features) == 0:
                counts.append(0)
                if bogie_data[bogie_id]['best_image'] is not None:
                    processed_images.append(bogie_data[bogie_id]['best_image'])
                    processed_image_paths.append(bogie_data[bogie_id]['all_image_paths'][0])
                continue
            
            # Cluster similar features (representing the same person)
            from sklearn.cluster import DBSCAN
            
            # Use DBSCAN to cluster similar features
            # Epsilon (max distance between samples) needs tuning based on your feature space
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(all_features)
            
            # Number of unique people is the number of clusters
            unique_people = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            # Create a combined visualization
            if bogie_data[bogie_id]['best_image'] is not None:
                best_img = bogie_data[bogie_id]['best_image'].copy()
                
                # Update the count text to show unique people
                cv2.putText(best_img, f"Bogie {bogie_id+1}: {unique_people} unique people", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(best_img, f"(from {len(bogie_data[bogie_id]['all_images'])} images)", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Determine crowd level based on unique count
                crowd_level = "Empty"
                for level, threshold in CROWD_LEVELS.items():
                    if unique_people < threshold:
                        crowd_level = level
                        break
                
                cv2.putText(best_img, f"Status: {crowd_level}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save the updated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("static/results", f"bogie_{bogie_id+1}_combined_{timestamp}.jpg")
                cv2.imwrite(output_path, best_img)
                
                processed_images.append(best_img)
                processed_image_paths.append(output_path.replace("static/", ""))
                counts.append(unique_people)
            else:
                counts.append(0)
        else:
            # For single image bogies, use the original count
            counts.append(bogie_data[bogie_id]['best_count'])
            if bogie_data[bogie_id]['best_image'] is not None:
                processed_images.append(bogie_data[bogie_id]['best_image'])
                processed_image_paths.append(bogie_data[bogie_id]['all_image_paths'][0])
    
    # Calculate redirection recommendations
    recommendations = calculate_redirections(counts)
    
    # Create summary image with recommendations
    if processed_images:
        summary = create_summary_image(processed_images, counts, recommendations)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join("static/results", f"summary_{timestamp}.jpg")
        cv2.imwrite(summary_path, summary)
        summary_path = summary_path.replace("static/", "")
    else:
        summary_path = ""
    
    # Get enhanced analysis from Groq with image paths for qualitative assessment
    groq_analysis = analyze_with_groq(counts, recommendations, None, image_paths)
    
    # Extract Groq's estimated counts from the analysis
    groq_counts = extract_groq_counts(groq_analysis, counts)
    
    # Store the original YOLO counts before any density-based corrections
    yolo_raw_counts = [bogie_data[bogie_id]['best_count'] for bogie_id in sorted(bogie_data.keys())]
    
    return {
        "counts": counts,  # These are the density-corrected counts
        "yolo_counts": yolo_raw_counts,  # Raw YOLO detection counts
        "groq_counts": groq_counts,  # LLM-estimated counts
        "recommendations": recommendations,
        "processed_images": processed_image_paths,
        "summary_image": summary_path,
        "groq_analysis": groq_analysis
    }

def calculate_redirections(counts):
    """Calculate how to redistribute passengers to balance the crowd"""
    if not counts:
        return "No images processed."
        
    total_people = sum(counts)
    ideal_per_bogie = total_people / len(counts)
    
    recommendations = []
    
    for i, count in enumerate(counts):
        if count > ideal_per_bogie * 1.2:  # More than 20% over ideal
            # Find the least crowded bogie
            min_idx = counts.index(min(counts))
            move_count = int(count - ideal_per_bogie)
            recommendations.append(f"Move approximately {move_count} people from Bogie {i+1} to Bogie {min_idx+1}")
    
    if not recommendations:
        recommendations.append("Crowd is relatively balanced across all bogies.")
    
    return "\n".join(recommendations)

def create_summary_image(images, counts, recommendations):
    """Create a summary image showing all bogies and recommendations"""
    # Ensure all images have the same dimensions
    h, w = 720, 1280
    resized_images = []
    for img in images:
        resized = cv2.resize(img, (w, h))
        resized_images.append(resized)
    
    # Pad with blank images if needed
    while len(resized_images) < 6:
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        resized_images.append(blank)
        counts.append(0)
    
    # Create a blank canvas
    summary = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # Place the 6 images in a 2x3 grid
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for i, img in enumerate(resized_images[:6]):
        row, col = positions[i]
        summary[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # Add color indicators based on crowd density
    for i, count in enumerate(counts[:6]):
        row, col = positions[i]
        # Add colored rectangle indicator
        if count <= 10:  # Empty
            color = (255, 0, 0)  # Blue
        elif count <= 20:  # A few people
            color = (0, 255, 0)  # Green
        elif count <= 40:  # Average crowd
            color = (0, 255, 255)  # Yellow
        elif count <= 60:  # Getting busy
            color = (0, 165, 255)  # Orange
        elif count <= 80:  # Busy
            color = (0, 0, 255)  # Red
        else:  # Packed
            color = (0, 0, 128)  # Dark Red
        
        cv2.rectangle(summary, (col*w, row*h), (col*w+50, row*h+50), color, -1)
    
    # Add recommendations text
    y_offset = h*2 - 150
    for line in recommendations.split('\n'):
        cv2.putText(summary, line, (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
    
    return summary

def analyze_platform_image(image_path):
    """Analyze platform image using the enhanced dense crowd detection algorithm"""
    # Import the YOLO model if not already imported
    if 'yolo_model' not in globals():
        from ultralytics import YOLO
        global yolo_model
        yolo_model = YOLO("Yolo-Weights/yolov8l.pt")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # First try YOLO with very low confidence to get some detections
    results = yolo_model(img, conf=0.05)  # Even lower threshold for extremely dense crowds
    
    # Process YOLO results
    yolo_count = 0
    person_boxes = []
    
    # Extract all person detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if yolo_model.names[cls] == "person" and conf > 0.05:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                yolo_count += 1
                person_boxes.append((x1, y1, x2, y2))
    
    # Add right side detection code
    img_height, img_width = img.shape[:2]
    
    # Create a right-side region (right 40% of the image)
    right_region = img[:, int(img_width * 0.6):]
    
    # Run detection specifically on this region with even lower confidence
    right_results = yolo_model(right_region, conf=0.03)
    
    # Process right side detections and adjust coordinates
    for r in right_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if yolo_model.names[cls] == "person" and conf > 0.03:
                # Adjust coordinates to match the original image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += int(img_width * 0.6)  # Adjust x-coordinates
                x2 += int(img_width * 0.6)
                
                # Check if this detection overlaps significantly with existing ones
                overlap = False
                for existing_box in person_boxes:
                    ex1, ey1, ex2, ey2 = existing_box
                    # Calculate IoU (Intersection over Union)
                    x_left = max(x1, ex1)
                    y_top = max(y1, ey1)
                    x_right = min(x2, ex2)
                    y_bottom = min(y2, ey2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (ex2 - ex1) * (ey2 - ey1)
                        union = area1 + area2 - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:  # If overlap is significant
                            overlap = True
                            break
                
                # If no significant overlap, add this as a new detection
                if not overlap:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    yolo_count += 1
                    person_boxes.append((x1, y1, x2, y2))
    
    # If we have some detections, use them to estimate crowd density
    if len(person_boxes) > 0:
        # Calculate average person size
        avg_width = sum(x2-x1 for x1,y1,x2,y2 in person_boxes) / len(person_boxes)
        avg_height = sum(y2-y1 for x1,y1,x2,y2 in person_boxes) / len(person_boxes)
        avg_area = avg_width * avg_height
        
        # Calculate total platform area (excluding tracks)
        platform_area = img.shape[0] * img.shape[1] * 0.7
        
        # In extremely dense crowds, people stand much closer
        spacing_factor = 1.2  # Very dense crowds have less personal space
        
        # Estimate maximum capacity
        max_capacity = platform_area / (avg_area * spacing_factor)
        
        # Use color variance as a crowd density indicator
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_variance = np.std(hsv[:,:,0])
        
        # Calculate edge density (more edges often means more people)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # Combine multiple indicators for better estimation
        density_score = 0.0
        
        # If color variance is high (many different colored clothing)
        if color_variance > 40:  # Threshold determined empirically
            density_score += 0.3
        
        # If edge density is high (many edges between people)
        if edge_density > 0.1:  # Threshold determined empirically
            density_score += 0.3
        
        # If we detected some people but not many (occlusion issue)
        if len(person_boxes) > 5 and len(person_boxes) < 50:
            density_score += 0.2
        
        # Calculate estimated count based on density score
        if density_score > 0.6:  # High density indicators
            # For very high density, use 80-90% of max capacity
            estimated_count = int(max_capacity * 0.85)
        elif density_score > 0.4:  # Medium-high density
            estimated_count = int(max_capacity * 0.6)
        elif density_score > 0.2:  # Medium density
            estimated_count = int(max_capacity * 0.4)
        else:  # Low density, trust YOLO count
            estimated_count = yolo_count
        
        # Use the higher of YOLO count or estimated count
        final_count = max(yolo_count, estimated_count)
        
        # Create visualization
        # Create a heatmap based on edge density and color variance
        heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        # Add heat based on edge density
        edge_contribution = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (21, 21), 0)
        heatmap += edge_contribution * 0.5
        
        # Add heat based on detected people
        for x1, y1, x2, y2 in person_boxes:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            person_size = max(x2-x1, y2-y1)
            sigma = person_size
            y_grid, x_grid = np.ogrid[-center_y:img.shape[0]-center_y, -center_x:img.shape[1]-center_x]
            mask = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2*sigma*sigma))
            heatmap += mask
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert heatmap to color image
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        result = cv2.addWeighted(img, 0.7, heatmap_color, 0.3, 0)
        
        # Determine density level for train frequency
        if final_count <= 10:
            density_level = "low"
            color = (255, 0, 0)  # Blue
        elif final_count <= 30:
            density_level = "medium"
            color = (0, 255, 0)  # Green
        else:
            density_level = "high"
            color = (0, 0, 255)  # Red
        
        # Add information to image
        cv2.putText(result, f"Dense Crowd Count: {final_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result, f"YOLO Count: {yolo_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result, f"Density Score: {density_score:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, f"Status: {'Packed' if final_count > 80 else 'Busy'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Get train frequency from TRAIN_FREQUENCY dictionary
        train_frequency = TRAIN_FREQUENCY[density_level]
        cv2.putText(result, f"Recommended Train Frequency: {train_frequency} min", 
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add density indicator (red for high density)
        cv2.rectangle(result, (10, 200), (60, 250), color, -1)
        
        # Save the processed image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("static/results", f"platform_{timestamp}.jpg")
        cv2.imwrite(output_path, result)
        
        return {
            "total_people": final_count,
            "yolo_count": yolo_count,
            "density_score": density_score,
            "density_level": density_level,
            "train_frequency": train_frequency,
            "processed_image": output_path.replace("static/", "")
        }
    else:
        # If no people detected, return basic info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("static/results", f"platform_{timestamp}.jpg")
        cv2.imwrite(output_path, img)
        
        return {
            "total_people": 0,
            "yolo_count": 0,
            "density_score": 0.0,
            "density_level": "low",
            "train_frequency": TRAIN_FREQUENCY["low"],
            "processed_image": output_path.replace("static/", "")
        }

# Remove the @app.route('/') from the analyze function

# Add this new route handler for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Then keep your existing analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    # Add debug logging
    print("Request files keys:", list(request.files.keys()))
    print("Request form keys:", list(request.form.keys()))
    
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
            print("Error converting bogie IDs to integers")
            # Fall back to default IDs
            bogie_ids = list(range(len(files)))
    
    # Save uploaded files
    image_paths = []
    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            image_paths.append(filepath)
    
    if not image_paths:
        return jsonify({'error': 'No valid image files uploaded'})
    
    # Process platform image if provided
    platform_data = None
    if 'platform_image' in request.files:
        platform_file = request.files['platform_image']
        if platform_file and platform_file.filename != '':
            platform_filename = secure_filename(platform_file.filename)
            platform_filepath = os.path.join('uploads/platforms', platform_filename)
            platform_file.save(platform_filepath)
            platform_data = analyze_platform_image(platform_filepath)
    
    # Process metro images with bogie IDs
    results = process_metro_images(image_paths, bogie_ids)
    
    # Add platform data to results
    if platform_data:
        results['platform_data'] = platform_data
    
    return jsonify(results)

@app.route('/analyze_platform', methods=['POST'])
def analyze_platform():
    if 'platform_image' not in request.files:
        return jsonify({'error': 'No platform image uploaded'})
    
    platform_file = request.files['platform_image']
    
    if platform_file.filename == '':
        return jsonify({'error': 'No platform image selected'})
    
    # Save uploaded file
    platform_path = os.path.join('uploads/platforms', 'platform.jpg')
    platform_file.save(platform_path)
    
    # Analyze platform image
    platform_data = analyze_platform_image(platform_path)
    
    if not platform_data:
        return jsonify({'error': 'Failed to analyze platform image'})
    
    return jsonify(platform_data)

if __name__ == '__main__':
    app.run(debug=True)
    
    # Remove or comment out all these lines:
    # print("YOLO counts:", counts)
    # 
    # If you're using ResNet18 model anywhere, add this:
    # for i, img_path in enumerate(image_paths):
    #     model = load_crowd_counter_model()
    #     resnet_count, _ = predict_crowd(model, img_path)
    #     print(f"ResNet count for bogie {i+1}:", resnet_count)
    #     # Add specialized head detection for crowded scenes
    #     # This is particularly effective when bodies are occluded but heads are visible
    #     head_results = yolo_model(img, conf=0.08)  # Very low threshold specifically for head detection
    #     
    #     # Process head detections
    #     for r in head_results:
    #         for box in r.boxes:
    #             cls = int(box.cls[0])
    #             conf = box.conf[0]
    #             if yolo_model.names[cls] == "person" and conf > 0.08:
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 
    #                 # Focus only on potential heads (small boxes in upper part of image or seats)
    #                 box_area = (x2-x1) * (y2-y1)
    #                 img_area = img.shape[0] * img.shape[1]
    #                 
    #                 # Head boxes are typically small and in upper part of the image
    #                 if box_area < (img_area * 0.01) and y1 < (img.shape[0] * 0.7):  # Small box, likely a head
    #                     # Check if this is a new detection (not overlapping with existing ones)
    #                     new_detection = True
    #                     for ex1, ey1, ex2, ey2 in person_boxes:
    #                         # Calculate overlap
    #                         overlap_x1 = max(x1, ex1)
    #                         overlap_y1 = max(y1, ey1)
    #                         overlap_x2 = min(x2, ex2)
    #                         overlap_y2 = min(y2, ey2)
    #                         
    #                         if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
    #                             overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    #                             head_area = (x2 - x1) * (y2 - y1)
    #                             
    #                             if overlap_area / head_area > 0.3:  # Lower threshold for heads
    #                                 new_detection = False
    #                                 break
    #                     
    #                     if new_detection:
    #                         cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for heads
    #                         cv2.putText(img_display, "H", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    #                         yolo_count += 1
    #                         person_boxes.append((x1, y1, x2, y2))
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Add this near the top with other app configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit


def generate_fallback_analysis(counts, recommendations, platform_data=None):
    """Generate a basic analysis when the API fails"""
    total_passengers = sum(counts)
    avg_per_bogie = total_passengers / len(counts) if counts else 0
    most_crowded = counts.index(max(counts)) + 1 if counts else 0
    least_crowded = counts.index(min(counts)) + 1 if counts else 0
    
    analysis = """## Metro Crowd Analysis (Basic)

### Current Situation
"""
    analysis += f"Total passengers: {total_passengers}\n"
    analysis += f"Average passengers per bogie: {avg_per_bogie:.1f}\n"
    analysis += f"Most crowded bogie: {most_crowded} ({max(counts) if counts else 0} passengers)\n"
    analysis += f"Least crowded bogie: {least_crowded} ({min(counts) if counts else 0} passengers)\n\n"
    
    # Add basic recommendations
    analysis += "### Recommendations\n\n"
    analysis += "* Consider redistributing passengers from the most crowded bogie\n"
    analysis += "* Monitor the situation in real-time for changes\n"
    analysis += "* Ensure all available doors are operational for efficient boarding/alighting\n"
    
    return analysis


def extract_groq_counts(analysis, default_counts):
    """Extract the estimated counts from Groq's analysis"""
    try:
        # Look for the table with estimated counts
        import re
        
        # Pattern to match the table rows with counts
        pattern = r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        
        # Find all matches
        matches = re.findall(pattern, analysis)
        
        if matches:
            # Extract the estimated counts (third column)
            groq_counts = [int(match[2]) for match in matches]
            return groq_counts
        else:
            # If no table found, return the default counts
            return default_counts
    except Exception as e:
        print(f"Error extracting Groq counts: {e}")
        return default_counts

def validate_groq_analysis(analysis, counts):
    """Validate and enhance the Groq analysis to ensure quality"""
    # Check if analysis contains required sections
    required_sections = [
        "crowd distribution", "bottlenecks", "redistribution", 
        "prediction", "improving passenger comfort"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in analysis.lower():
            missing_sections.append(section)
    
    # If sections are missing, add placeholders
    if missing_sections:
        analysis += "\n\n## Additional Analysis\n\n"
        for section in missing_sections:
            analysis += f"### {section.title()}\n\n"
            analysis += f"Based on the data, the {section} analysis indicates...[standard recommendation]\n\n"
    
    # Add factual data summary at the beginning
    summary = "## Summary of Crowd Data\n\n"
    summary += "| Bogie | Passenger Count | Density Level |\n"
    summary += "|-------|-----------------|---------------|\n"
    
    for i, count in enumerate(counts):
        density = "Low" if count < 30 else "Medium" if count < 60 else "High"
        summary += f"| {i+1} | {count} | {density} |\n"
    
    return summary + "\n\n" + analysis