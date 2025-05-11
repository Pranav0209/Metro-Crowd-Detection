import os
import cv2
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN

# Define crowd density levels for inference
CROWD_LEVELS = {
    "Empty": 10,
    "A few people": 20,
    "Average crowd": 40,
    "Getting busy": 60,
    "Busy": 80,
    "Packed": float('inf')
}

# Train frequency settings
TRAIN_FREQUENCY = {
    "low": 15,     # 15 minutes between trains during low platform crowd
    "medium": 10,  # 10 minutes between trains during medium platform crowd
    "high": 5      # 5 minutes between trains during high platform crowd
}

def calculate_redirections(counts):
    """Calculate how to redistribute passengers to balance the crowd
    
    Args:
        counts: List of people counts for each bogie
        
    Returns:
        recommendations: Recommendations for crowd redistribution
    """
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
    """Create a summary image showing all bogies and recommendations
    
    Args:
        images: List of processed images
        counts: List of people counts for each bogie
        recommendations: Recommendations for crowd redistribution
        
    Returns:
        summary: Summary image
    """
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

def match_people_across_images(bogie_data):
    """Match people across multiple images of the same bogie to avoid double counting
    
    Args:
        bogie_data: Dictionary containing person features and boxes by bogie ID
        
    Returns:
        counts: List of unique people counts for each bogie
        processed_images: List of processed images
        processed_image_paths: List of processed image paths
        yolo_counts: List of YOLO counts for each bogie
        density_counts: List of density-based counts for each bogie
        groq_counts: List of Groq LLM counts for each bogie
        standing_counts: List of standing people counts for each bogie
        sitting_counts: List of sitting people counts for each bogie
        adjusted_sitting_counts: List of adjusted sitting counts (accounting for all seats)
        all_seats_occupied_flags: List of flags indicating if all seats are occupied
    """
    counts = []
    processed_images = []
    processed_image_paths = []
    yolo_counts = []
    density_counts = []
    groq_counts = []
    standing_counts = []
    sitting_counts = []
    adjusted_sitting_counts = []
    all_seats_occupied_flags = []
    
    for bogie_id in sorted(bogie_data.keys()):
        # Extract model-specific counts
        yolo_count = bogie_data[bogie_id].get('yolo_count', 0)
        density_count = bogie_data[bogie_id].get('density_count', 0)
        groq_count = bogie_data[bogie_id].get('groq_count', 0)
        standing_count = bogie_data[bogie_id].get('standing_count', 0)
        sitting_count = bogie_data[bogie_id].get('sitting_count', 0)
        adjusted_sitting_count = bogie_data[bogie_id].get('adjusted_sitting_count', sitting_count)
        all_seats_occupied = bogie_data[bogie_id].get('all_seats_occupied', False)
        
        # Store model-specific counts
        yolo_counts.append(yolo_count)
        density_counts.append(density_count)
        groq_counts.append(groq_count)
        standing_counts.append(standing_count)
        sitting_counts.append(sitting_count)
        adjusted_sitting_counts.append(adjusted_sitting_count)
        all_seats_occupied_flags.append(all_seats_occupied)
        
        # If we have multiple images for this bogie, perform matching
        if len(bogie_data[bogie_id]['all_images']) > 1:
            # Get all features from this bogie
            all_features = np.array(bogie_data[bogie_id]['person_features'])
            
            # Skip if no features
            if len(all_features) == 0:
                counts.append(0)
                yolo_counts.append(yolo_count)
                density_counts.append(density_count)
                groq_counts.append(groq_count)
                if bogie_data[bogie_id]['best_image'] is not None:
                    processed_images.append(bogie_data[bogie_id]['best_image'])
                    processed_image_paths.append(bogie_data[bogie_id]['all_image_paths'][0])
                continue
            
            # Use DBSCAN to cluster similar features
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(all_features)
            
            # Number of unique people is the number of clusters
            unique_people = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            # Create a combined visualization
            if bogie_data[bogie_id]['best_image'] is not None:
                best_img = bogie_data[bogie_id]['best_image'].copy()
                
                # Update the count text to show unique people and all model results
                cv2.putText(best_img, f"Bogie {bogie_id+1}: Analysis Results", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(best_img, f"YOLO: {yolo_count} | Density: {density_count} | Groq: {groq_count}", 
                # Add model-specific counts
                cv2.putText(best_img, f"YOLO: {yolo_count} | Density: {density_count} | Groq: {groq_count}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(best_img, f"Combined: {unique_people} unique people", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(best_img, f"(from {len(bogie_data[bogie_id]['all_images'])} images)", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Determine crowd level based on unique count
                crowd_level = "Empty"
                for level, threshold in CROWD_LEVELS.items():
                    if unique_people < threshold:
                        crowd_level = level
                        break
                
                cv2.putText(best_img, f"Status: {crowd_level}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save the updated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("static/results", f"bogie_{bogie_id+1}_combined_{timestamp}.jpg")
                cv2.imwrite(output_path, best_img)
                
                processed_images.append(best_img)
                processed_image_paths.append(output_path)
                counts.append(unique_people)
                yolo_counts.append(yolo_count)
                density_counts.append(density_count)
                groq_counts.append(groq_count)
            else:
                counts.append(0)
                yolo_counts.append(0)
                density_counts.append(0)
                groq_counts.append(0)
        else:
            # For single image bogies, use the original count
            counts.append(bogie_data[bogie_id]['best_count'])
            yolo_counts.append(yolo_count)
            density_counts.append(density_count)
            groq_counts.append(groq_count)
            if bogie_data[bogie_id]['best_image'] is not None:
                processed_images.append(bogie_data[bogie_id]['best_image'])
                processed_image_paths.append(bogie_data[bogie_id]['all_image_paths'][0])
    
    return counts, processed_images, processed_image_paths, yolo_counts, density_counts, groq_counts, standing_counts, sitting_counts, adjusted_sitting_counts, all_seats_occupied_flags

def allowed_file(filename):
    """Check if a file has an allowed extension
    
    Args:
        filename: Name of the file
        
    Returns:
        allowed: Whether the file has an allowed extension
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS