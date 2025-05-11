import os
import cv2
import numpy as np
from datetime import datetime
from models.utils import TRAIN_FREQUENCY

class PlatformAnalyzer:
    def __init__(self, yolo_detector):
        """Initialize platform analyzer
        
        Args:
            yolo_detector: YOLO detector instance
        """
        self.yolo_detector = yolo_detector
    
    def analyze_platform_image(self, image_path):
        """Analyze platform image using the enhanced dense crowd detection algorithm
        
        Args:
            image_path: Path to the platform image
            
        Returns:
            platform_data: Dictionary containing platform analysis results
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # First try YOLO with very low confidence to get some detections
        conf_threshold = 0.05  # Even lower threshold for extremely dense crowds
        results = self.yolo_detector.model(img, conf=conf_threshold)
        
        # Process YOLO results
        yolo_count = 0
        person_boxes = []
        
        # Extract all person detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                box_conf = box.conf[0]
                if self.yolo_detector.model.names[cls] == "person" and box_conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    yolo_count += 1
                    person_boxes.append((x1, y1, x2, y2))
        
        # Add right side detection code
        img_height, img_width = img.shape[:2]
        
        # Create a right-side region (right 40% of the image)
        right_region = img[:, int(img_width * 0.6):]
        
        # Run detection specifically on this region with even lower confidence
        right_conf_threshold = 0.03
        right_results = self.yolo_detector.model(right_region, conf=right_conf_threshold)
        
        # Process right side detections and adjust coordinates
        for r in right_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                box_conf = box.conf[0]
                if self.yolo_detector.model.names[cls] == "person" and box_conf > right_conf_threshold:
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
                "processed_image": output_path
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
                "processed_image": output_path
            }