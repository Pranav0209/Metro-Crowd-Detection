import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights_path="Yolo-Weights/yolov8l.pt"):
        """Initialize YOLO model for object detection
        
        Args:
            weights_path: Path to the YOLO weights file
        """
        self.model = YOLO(weights_path)
    
    def detect_people(self, img, conf_threshold=0.15):
        """Detect people in an image using YOLO
        
        Args:
            img: Input image (numpy array)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            results: YOLO detection results
            person_boxes: List of bounding boxes for detected people
            person_features: List of features for detected people
            yolo_count: Number of people detected
            img_display: Image with bounding boxes drawn
        """
        print(f"YOLO detect_people called with conf_threshold={conf_threshold}")
        
        # Create a copy for visualization
        img_display = img.copy()
        
        # YOLO detection with lower confidence threshold for seated passengers
        print("Running YOLO model...")
        results = self.model(img, conf=conf_threshold)
        print(f"YOLO results: {len(results)} detection batches")
        
        # Process YOLO results
        yolo_count = 0
        person_boxes = []
        person_features = []
        
        # Extract all person detections
        for r in results:
            print(f"Processing batch with {len(r.boxes)} detections")
            for box in r.boxes:
                cls = int(box.cls[0])
                box_conf = box.conf[0]
                print(f"Detection: class={cls} ({self.model.names[cls]}), confidence={box_conf:.2f}")
                if self.model.names[cls] == "person" and box_conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract person region for feature calculation
                    person_roi = img[y1:y2, x1:x2]
                    if person_roi.size == 0:  # Skip if ROI is empty
                        continue
                        
                    # Calculate simple feature vector (color histogram)
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
                        cv2.putText(img_display, f"{box_conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        yolo_count += 1
                    except Exception as e:
                        print(f"Error processing person ROI: {e}")
        
        return results, person_boxes, person_features, yolo_count, img_display
    
    def detect_seated_people(self, img, person_boxes, conf_threshold=0.08):
        """Detect seated people in an image using YOLO with special criteria
        
        Args:
            img: Input image (numpy array)
            person_boxes: Existing person boxes to avoid duplicates
            conf_threshold: Confidence threshold for detection (lowered for seated people)
            
        Returns:
            new_person_boxes: List of new bounding boxes for detected seated people
            new_person_features: List of features for detected seated people
            seated_count: Number of seated people detected
            img_display: Updated image with bounding boxes drawn
        """
        img_display = img.copy()
        seated_results = self.model(img, conf=conf_threshold)
        
        new_person_boxes = []
        new_person_features = []
        seated_count = 0
        
        # Process seated detections with different criteria
        for r in seated_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                box_conf = box.conf[0]
                if self.model.names[cls] == "person" and box_conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Check if this is likely a seated person (wider than tall, or in seat position)
                    width, height = x2-x1, y2-y1
                    aspect_ratio = width/height if height > 0 else 0
                    
                    # Enhanced seated people detection with more permissive criteria
                    # Seated people often have different aspect ratios or are partially visible
                    if aspect_ratio > 0.6 or y1 > img.shape[0]/3 or height < img.shape[0]/4:
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
                                    new_person_features.append(feature)
                                    new_person_boxes.append((x1, y1, x2, y2))
                                    
                                    # Draw bounding box with different color for seated people
                                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                    seated_count += 1
                            except Exception as e:
                                print(f"Error processing seated person ROI: {e}")
        
        return new_person_boxes, new_person_features, seated_count, img_display
    
    def apply_density_correction(self, img, person_boxes, yolo_count):
        """Apply density-based correction to the people count with enhanced crowding detection
        
        Args:
            img: Input image (numpy array)
            person_boxes: List of bounding boxes for detected people
            yolo_count: Initial count from YOLO detection
            
        Returns:
            corrected_count: Corrected people count
        """
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
            
            # Calculate texture complexity (higher in crowded scenes)
            texture_complexity = np.var(gray)
            
            # Calculate person area ratio (how much of the image is covered by detected people)
            person_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in person_boxes)
            visible_area = img.shape[0] * img.shape[1] * 0.6  # Assume 60% of image is usable space
            area_ratio = person_area / visible_area
            
            # Check for potential occlusions (people behind people)
            # In crowded scenes, there's often more overlap between bounding boxes
            overlap_count = 0
            for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                for j, (x1b, y1b, x2b, y2b) in enumerate(person_boxes):
                    if i != j:
                        # Calculate overlap
                        overlap_x1 = max(x1, x1b)
                        overlap_y1 = max(y1, y1b)
                        overlap_x2 = min(x2, x2b)
                        overlap_y2 = min(y2, y2b)
                        
                        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            box_area = (x2 - x1) * (y2 - y1)
                            
                            if overlap_area / box_area > 0.3:  # Significant overlap
                                overlap_count += 1
            
            # Normalize overlap count
            overlap_ratio = overlap_count / max(1, len(person_boxes))
            
            # Enhanced correction for crowded scenes
            corrected_count = yolo_count
            
            # Combine multiple indicators for crowding detection
            crowding_score = (
                edge_density * 5 + 
                color_variance / 50 + 
                area_ratio * 3 + 
                overlap_ratio * 2 +
                texture_complexity / 1000
            )
            
            print(f"Crowding indicators - Edge: {edge_density:.3f}, Color: {color_variance:.1f}, " +
                  f"Area: {area_ratio:.3f}, Overlap: {overlap_ratio:.3f}, Texture: {texture_complexity:.1f}")
            print(f"Combined crowding score: {crowding_score:.3f}")
            
            # Apply correction based on crowding score
            if crowding_score > 2.0:  # Very crowded
                # Use tighter spacing factor for crowded bogies
                estimated_capacity = int(visible_area / (avg_width * avg_height * 1.0))
                corrected_count = max(yolo_count, int(estimated_capacity * 0.9))
                print(f"Applied HEAVY density correction: {yolo_count} → {corrected_count}")
            elif crowding_score > 1.5:  # Moderately crowded
                estimated_capacity = int(visible_area / (avg_width * avg_height * 1.2))
                corrected_count = max(yolo_count, int(estimated_capacity * 0.75))
                print(f"Applied MEDIUM density correction: {yolo_count} → {corrected_count}")
            elif crowding_score > 1.0:  # Slightly crowded
                estimated_capacity = int(visible_area / (avg_width * avg_height * 1.5))
                corrected_count = max(yolo_count, int(estimated_capacity * 0.6))
                print(f"Applied LIGHT density correction: {yolo_count} → {corrected_count}")
            
            return corrected_count
        
        return yolo_count
    
    def create_heatmap(self, img, person_boxes):
        """Create a heatmap visualization based on detected people
        
        Args:
            img: Input image (numpy array)
            person_boxes: List of bounding boxes for detected people
            
        Returns:
            overlay: Image with heatmap overlay
        """
        # Create a heatmap visualization
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
                
                # Add to heatmap
                heatmap = np.maximum(heatmap, mask)
            
            # Normalize heatmap
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply colormap
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Create overlay
            overlay = img.copy()
            alpha = 0.4
            cv2.addWeighted(heatmap_color, alpha, overlay, 1-alpha, 0, overlay)
            
            # Draw bounding boxes
            for x1, y1, x2, y2 in person_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return overlay
        
        return img.copy()