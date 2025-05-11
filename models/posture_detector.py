"""
Posture Detector Module

This module extends YOLO detection to classify people as either standing or sitting
based on their bounding box aspect ratio and position.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class PostureDetector:
    """Detects whether people are standing or sitting based on YOLO detections"""
    
    def __init__(self, yolo_model=None):
        """Initialize the posture detector
        
        Args:
            yolo_model: Optional pre-loaded YOLO model
        """
        self.yolo_model = yolo_model if yolo_model else YOLO('yolov8n.pt')
        
        # Parameters for posture classification - tuned for metro train images
        self.aspect_ratio_threshold = 1.8  # Standing people typically have higher aspect ratio
        self.position_threshold = 0.65  # Sitting people are typically in the lower part of the image
        self.height_threshold = 0.3  # Standing people typically occupy a larger portion of the image height
        self.width_threshold = 0.15  # Sitting people typically occupy more width
        
        # Aisle detection parameters
        self.aisle_center = 0.5  # Center of the aisle (normalized)
        self.aisle_width = 0.25  # Width of the aisle (normalized)
        
        # Confidence thresholds
        self.min_confidence = 0.35  # Minimum confidence for detection
        
    def detect_postures(self, image, seats_per_bogie=43):
        """Detect people and classify their posture
        
        Args:
            image: Input image (numpy array)
            seats_per_bogie: Number of seats in a standard metro bogie
            
        Returns:
            results: Dictionary with detection results including posture
        """
        # Get original image dimensions
        img_height, img_width = image.shape[:2]
        
        # Run YOLO detection
        detections = self.yolo_model(image, classes=0)  # Class 0 is person
        
        # Process results
        boxes = []
        standing_count = 0
        sitting_count = 0
        
        # Extract detection results
        for detection in detections:
            for box in detection.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Skip low confidence detections
                if confidence < self.min_confidence:
                    continue
                
                # Calculate basic measurements
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width
                
                # Calculate relative positions and sizes
                relative_width = width / img_width
                relative_height = height / img_height
                relative_top = y1 / img_height
                relative_bottom = y2 / img_height
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                
                # Determine if person is in the aisle
                aisle_left = self.aisle_center - (self.aisle_width / 2)
                aisle_right = self.aisle_center + (self.aisle_width / 2)
                is_in_aisle = aisle_left < center_x < aisle_right
                
                # Initialize posture classification variables
                posture = "unknown"
                posture_confidence = 0.0
                notes = []
                
                # ===== POSTURE CLASSIFICATION LOGIC =====
                
                # 1. Strong indicators for STANDING
                strong_standing_indicators = 0
                
                # 1a. Tall aspect ratio (height significantly greater than width)
                if aspect_ratio > self.aspect_ratio_threshold + 0.3:
                    strong_standing_indicators += 1
                    notes.append("tall_ratio")
                
                # 1b. Person occupies significant height of the image
                if relative_height > self.height_threshold + 0.1:
                    strong_standing_indicators += 1
                    notes.append("tall")
                
                # 1c. Person is in the aisle and reasonably tall
                if is_in_aisle and relative_height > 0.25:
                    strong_standing_indicators += 1
                    notes.append("aisle")
                
                # 1d. Person's top is in the upper third of the image
                if relative_top < 0.33:
                    strong_standing_indicators += 1
                    notes.append("upper_body")
                
                # 2. Strong indicators for SITTING
                strong_sitting_indicators = 0
                
                # 2a. Wide aspect ratio (width greater than height)
                if aspect_ratio < 1.0:
                    strong_sitting_indicators += 1
                    notes.append("wide")
                
                # 2b. Person is positioned in the bottom part of the image
                if relative_bottom > 0.85 and relative_top > 0.5:
                    strong_sitting_indicators += 1
                    notes.append("bottom_position")
                
                # 2c. Person is wide relative to the image
                if relative_width > self.width_threshold and relative_height < 0.25:
                    strong_sitting_indicators += 1
                    notes.append("wide_area")
                
                # 3. Make the final decision
                if strong_standing_indicators >= 2 and strong_standing_indicators > strong_sitting_indicators:
                    posture = "standing"
                    posture_confidence = min(0.9, 0.6 + (strong_standing_indicators * 0.1))
                elif strong_sitting_indicators >= 1 and strong_sitting_indicators >= strong_standing_indicators:
                    posture = "sitting"
                    posture_confidence = min(0.9, 0.6 + (strong_sitting_indicators * 0.1))
                else:
                    # Use aspect ratio as a fallback
                    if aspect_ratio > self.aspect_ratio_threshold:
                        posture = "standing"
                        posture_confidence = 0.6
                    else:
                        posture = "sitting"
                        posture_confidence = 0.6
                
                # Debug information
                debug_info = f"AR={aspect_ratio:.2f}, H={relative_height:.2f}, W={relative_width:.2f}, " + \
                             f"Y={relative_top:.2f}-{relative_bottom:.2f}, Aisle={is_in_aisle}, " + \
                             f"Stand={strong_standing_indicators}, Sit={strong_sitting_indicators}, " + \
                             f"Posture={posture} ({posture_confidence:.2f})"
                print(debug_info)
                
                # Update counts
                if posture == "sitting":
                    sitting_count += 1
                else:
                    standing_count += 1
                
                # Store detection with posture and additional info
                box_info = {
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'posture': posture,
                    'posture_confidence': float(posture_confidence),
                    'notes': ' '.join(notes)
                }
                
                boxes.append(box_info)
        
        # Apply the metro train logic with threshold for standing people
        adjusted_sitting_count = sitting_count
        all_seats_occupied = False
        standing_threshold = 3  # Minimum number of standing people to consider the train crowded
        
        # Only consider the train crowded if we detect a significant number of standing people
        if standing_count >= standing_threshold:
            # If we detect enough standing people, assume all seats are occupied
            all_seats_occupied = True
            # If detected sitting people are fewer than seats, adjust the count
            if sitting_count < seats_per_bogie:
                adjusted_sitting_count = seats_per_bogie
            
            # Calculate total count with the adjusted sitting count
            total_count = standing_count + adjusted_sitting_count
        else:
            # If few or no standing people, just use the actual detected counts
            # This means the train is not crowded enough to assume all seats are taken
            all_seats_occupied = False
            total_count = standing_count + sitting_count
        
        # Return results
        return {
            'boxes': boxes,
            'standing_count': standing_count,
            'sitting_count': sitting_count,
            'adjusted_sitting_count': adjusted_sitting_count,
            'all_seats_occupied': all_seats_occupied,
            'total_count': total_count
        }
    
    def visualize_postures(self, image, results):
        """Visualize posture detection results
        
        Args:
            image: Input image
            results: Detection results from detect_postures
            
        Returns:
            annotated_image: Image with posture annotations
        """
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Draw boxes and labels
        for box_info in results['boxes']:
            x1, y1, x2, y2 = box_info['box']
            posture = box_info['posture']
            confidence = box_info['confidence']
            
            # Set color based on posture (green for standing, blue for sitting)
            color = (0, 255, 0) if posture == "standing" else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence and notes
            posture_confidence = box_info.get('posture_confidence', 0.0)
            notes = box_info.get('notes', '')
            
            # Create a compact notes indicator
            notes_indicator = ""
            if "aisle" in notes:
                notes_indicator += " (A)"
            if "tall" in notes or "tall_ratio" in notes:
                notes_indicator += " (T)"
            if "upper_body" in notes:
                notes_indicator += " (U)"
            if "wide" in notes or "wide_area" in notes:
                notes_indicator += " (W)"
            
            # Create the label with detection confidence and posture confidence
            label = f"{posture}{notes_indicator} ({confidence:.2f}/{posture_confidence:.2f})"
            cv2.putText(annotated_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add summary with detected counts
        detected_summary = f"Detected - Standing: {results['standing_count']}, Sitting: {results['sitting_count']}"
        cv2.putText(annotated_image, detected_summary, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add adjusted counts if seats are assumed to be full
        if results.get('all_seats_occupied', False):
            # If standing count is significant, show the crowded train calculation
            adjusted_summary = f"Crowded Train - All {results['adjusted_sitting_count']} seats occupied + {results['standing_count']} standing"
            cv2.putText(annotated_image, adjusted_summary, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            total_summary = f"Total passengers: {results['total_count']} (Full capacity calculation)"
            cv2.putText(annotated_image, total_summary, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If standing count is low, show the actual detected count
            if results['standing_count'] > 0:
                status = f"Not Crowded - {results['standing_count']} standing (below threshold)"
                cv2.putText(annotated_image, status, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                status = f"Not Crowded - No standing passengers detected"
                cv2.putText(annotated_image, status, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            total_summary = f"Total passengers: {results['total_count']} (Actual count)"
            cv2.putText(annotated_image, total_summary, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_image
    
    def tune_parameters(self, aspect_ratio_threshold=None, position_threshold=None):
        """Tune the parameters for posture classification
        
        Args:
            aspect_ratio_threshold: New aspect ratio threshold
            position_threshold: New position threshold
        """
        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        
        if position_threshold is not None:
            self.position_threshold = position_threshold