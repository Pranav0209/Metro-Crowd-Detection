🚆 A. In Bogie Images
Found in process_metro_images(...)

Step 1: Visual Metrics Calculation
For each image:

avg_width, avg_height: bounding box averages

color_variance: from HSV hue channel → different clothes

edge_density: from Canny edges → clutter = people

area_ratio: total area covered by detected boxes / usable area

Step 2: Threshold Logic
If:

edge_density > 0.08 OR

color_variance > 25 OR

area_ratio > 0.15
AND person count < 20 → trigger correction

Step 3: Correction Formula
estimated_capacity = visible_area / (avg_width * avg_height * 1.2)

Based on how crowded it visually looks: