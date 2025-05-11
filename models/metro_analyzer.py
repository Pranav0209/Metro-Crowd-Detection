import os
import cv2
import numpy as np
from datetime import datetime
from models.utils import CROWD_LEVELS, match_people_across_images, calculate_redirections, create_summary_image

class MetroAnalyzer:
    def __init__(self, yolo_detector, groq_analyzer):
        """Initialize metro analyzer
        
        Args:
            yolo_detector: YOLO detector instance
            groq_analyzer: Groq analyzer instance
        """
        self.yolo_detector = yolo_detector
        self.groq_analyzer = groq_analyzer
    
    def process_metro_images(self, image_paths, bogie_ids=None):
        """Process metro images and return results using enhanced detection for bogies
        
        Args:
            image_paths: List of paths to bogie images
            bogie_ids: Optional list of bogie identifiers (same length as image_paths)
                      If provided, images with the same ID are considered from the same bogie
                      
        Returns:
            results: Dictionary containing analysis results
        """
        # If no bogie_ids provided, assume each image is a different bogie
        if bogie_ids is None:
            bogie_ids = list(range(len(image_paths)))
        
        # Dictionary to store person features by bogie ID
        bogie_data = {}
        
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
            print(f"Processing image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            print(f"Image shape: {img.shape}")
            
            # Detect people using YOLO
            print("Running YOLO detection...")
            _, person_boxes, person_features, yolo_count, img_display = self.yolo_detector.detect_people(img)
            print(f"YOLO detection found {yolo_count} people")
            
            # Detect seated people
            new_person_boxes, new_person_features, seated_count, img_display = self.yolo_detector.detect_seated_people(
                img, person_boxes
            )
            
            # Update counts and boxes
            yolo_count += seated_count
            person_boxes.extend(new_person_boxes)
            person_features.extend(new_person_features)
            
            # Apply density-based correction (Model 2)
            density_count = self.yolo_detector.apply_density_correction(img, person_boxes, yolo_count)
            
            # Get Groq LLM count for this specific image (Model 3)
            # We'll do this per image rather than for all images at once
            img_path_for_groq = [img_path]  # Create a list with just this image
            groq_analysis_single = self.groq_analyzer.analyze_crowd([yolo_count], "", None, img_path_for_groq)
            groq_count = self.groq_analyzer.extract_groq_counts(groq_analysis_single, [yolo_count])[0]
            
            # Create a heatmap visualization
            overlay = self.yolo_detector.create_heatmap(img, person_boxes)
            
            # Determine crowd level based on the average of all three models
            avg_count = (yolo_count + density_count + groq_count) / 3
            crowd_level = "Empty"
            for level, threshold in CROWD_LEVELS.items():
                if avg_count < threshold:
                    crowd_level = level
                    break
            
            # Add count text to image with all three model results
            cv2.putText(overlay, f"Bogie {bogie_id+1}: Analysis Results", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(overlay, f"YOLO: {yolo_count} | Density: {density_count} | Groq: {groq_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(overlay, f"Status: {crowd_level}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
            # Store all three model counts in the bogie data
            if 'yolo_count' not in bogie_data[bogie_id]:
                bogie_data[bogie_id]['yolo_count'] = yolo_count
            if 'density_count' not in bogie_data[bogie_id]:
                bogie_data[bogie_id]['density_count'] = density_count
            if 'groq_count' not in bogie_data[bogie_id]:
                bogie_data[bogie_id]['groq_count'] = groq_count
            
            # Save the processed image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("static/results", f"bogie_{bogie_id+1}_view{i+1}_{timestamp}.jpg")
            cv2.imwrite(output_path, overlay)
            
            # Store data for this bogie
            bogie_data[bogie_id]['person_features'].extend(person_features)
            bogie_data[bogie_id]['person_boxes'].extend(person_boxes)
            bogie_data[bogie_id]['all_images'].append(overlay)
            bogie_data[bogie_id]['all_image_paths'].append(output_path)
            
            # Update best image if this has more detections
            if density_count > bogie_data[bogie_id]['best_count']:
                bogie_data[bogie_id]['best_count'] = density_count
                bogie_data[bogie_id]['best_image'] = overlay
        
        # Second pass: match people across images of the same bogie to avoid double counting
        counts, processed_images, processed_image_paths, yolo_counts, density_counts, groq_counts = match_people_across_images(bogie_data)
        
        # Calculate redirection recommendations
        recommendations = calculate_redirections(counts)
        
        # Create summary image with recommendations
        if processed_images:
            summary = create_summary_image(processed_images, counts, recommendations)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join("static/results", f"summary_{timestamp}.jpg")
            cv2.imwrite(summary_path, summary)
        else:
            summary_path = ""
        
        # Get enhanced analysis from Groq with image paths for qualitative assessment
        groq_analysis = self.groq_analyzer.analyze_crowd(counts, recommendations, None, image_paths)
        
        # Extract Groq counts and update bogie_data
        groq_counts_from_analysis = self.groq_analyzer.extract_groq_counts(groq_analysis, counts)
        
        # Update bogie_data with Groq counts
        for i, bogie_id in enumerate(sorted(bogie_data.keys())):
            if i < len(groq_counts_from_analysis):
                bogie_data[bogie_id]['groq_count'] = groq_counts_from_analysis[i]
        
        return {
            "counts": counts,  # These are the combined counts after matching
            "yolo_counts": yolo_counts,  # YOLO model counts
            "density_counts": density_counts,  # Density-based model counts
            "groq_counts": groq_counts,  # Groq LLM model counts
            "recommendations": recommendations,
            "processed_images": processed_image_paths,
            "summary_image": summary_path,
            "groq_analysis": groq_analysis
        }