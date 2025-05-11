import requests
import json
import re
from datetime import datetime

class GroqAnalyzer:
    def __init__(self, api_key=None, api_url="https://api.groq.com/openai/v1/chat/completions"):
        """Initialize Groq API analyzer
        
        Args:
            api_key: Groq API key
            api_url: Groq API URL
        """
        self.api_key = api_key
        self.api_url = api_url
    
    def analyze_crowd(self, counts, recommendations, platform_data=None, image_paths=None):
        """Use Groq API to get enhanced analysis of crowd distribution with qualitative assessment
        
        Args:
            counts: List of people counts for each bogie
            recommendations: Recommendations for crowd redistribution
            platform_data: Optional data about platform crowd
            image_paths: Optional list of image paths
            
        Returns:
            analysis: Enhanced analysis from Groq API
        """
        if not self.api_key:
            return self.generate_fallback_analysis(counts, recommendations, platform_data)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Create a more detailed prompt with specific context and image descriptions
            prompt = f"""You are an advanced crowd analysis system for metro trains. Analyze the following metro train crowd data with precision and factual accuracy:
            
Metro Train Analysis:
- Specific train ID: Metro Line {datetime.now().strftime('%Y%m%d-%H%M')}
- Time of analysis: {datetime.now().strftime('%H:%M')}
- Total bogies analyzed: {len(counts)}
- YOLO model passenger counts by bogie: {counts}
- Current system recommendations: {recommendations}

Important context about metro train passenger counting:
- Each bogie has a maximum capacity of approximately 80-100 passengers
- Standing passengers require ~0.4 square meters of space for comfort
- Seated passengers are often undercounted by 15-20% in computer vision systems
- Rush hour threshold is typically 70% of maximum capacity
- Computer vision detection frequently misses:
  * Seated passengers (especially those partially visible)
  * Passengers in crowded areas due to occlusion
  * Passengers in low-contrast clothing or in shadows
  * Children and shorter individuals
- Typical metro bogies have 40-50 seats, with remaining capacity for standing passengers
- In crowded scenarios, computer vision systems typically undercount by 25-40%
- In moderate crowds, undercount is typically 15-25%
- In light crowds, undercount is typically 5-15%
"""

            # Add platform data if available
            if platform_data:
                prompt += f"""

Platform crowd data:
- Total people waiting: {platform_data['total_people']}
- Platform crowd density: {platform_data['density_level']}
- Current train frequency: Every {platform_data['train_frequency']} minutes
"""

            # Add image descriptions if available
            if image_paths:
                prompt += """

Image Analysis Context:
The following images have been analyzed by computer vision systems:
"""
                for i, path in enumerate(image_paths):
                    bogie_id = i + 1
                    prompt += f"""
Image {i+1}: Bogie {bogie_id}
- This image shows the interior of metro bogie {bogie_id}
- The computer vision system detected {counts[i] if i < len(counts) else 'unknown'} passengers
- The image may contain seated passengers, standing passengers, and empty spaces
- Typical metro bogie layout includes rows of seats along the sides and standing space in the middle
- Passengers may be partially visible or occluded by others in crowded areas
"""

            # Add enhanced count estimation request
            prompt += """

Based on the data provided and your expertise in metro train crowd analysis, provide a comprehensive analysis:

## Part 1: Passenger Count Estimation

1. For each bogie, provide a more accurate passenger count estimate by considering:
   - The computer vision count provided
   - Likely seated passengers that might be missed (especially in window seats)
   - Partially occluded passengers in crowded areas
   - Passengers in shadows or with low contrast against backgrounds
   - Children or shorter individuals who may be missed
   - The typical undercount percentages based on crowd density

2. Present your count estimates in this format:
   | Bogie | Computer Vision Count | Your Estimated Count | Confidence (1-10) | Reasoning |
   |-------|----------------------|---------------------|-------------------|-----------|
   | 1     | [count]              | [your estimate]     | [confidence]      | [brief explanation] |

## Part 2: Detailed Crowd Analysis

1. **Computer Vision Accuracy Assessment**:
   - Evaluate if the algorithmic counts seem accurate or if they might be underestimating the actual crowd
   - Identify specific scenarios where computer vision might be failing (e.g., seated passengers, crowded areas)
   - Estimate the overall accuracy of the computer vision system for this specific set of images

2. **Passenger Distribution Analysis**:
   - Analyze how passengers are distributed across all bogies
   - Identify which bogies are most crowded and which have available capacity
   - Calculate the standard deviation of passenger counts to assess balance

3. **Comfort and Density Analysis**:
   - Estimate the perceived crowding level from a passenger comfort perspective for each bogie
   - Classify each bogie as "Comfortable", "Moderately Crowded", or "Overcrowded"
   - Estimate the standing density (passengers per square meter) in crowded areas

4. **Hidden Passenger Assessment**:
   - Identify bogies that likely have hidden or occluded passengers not captured in the counts
   - Estimate the percentage of passengers that might be missed in each bogie
   - Suggest specific areas within bogies where passengers might be missed

## Part 3: Crowd Balancing and Redistribution Plan

1. **Optimal Passenger Distribution Plan**:
   - Calculate the ideal number of passengers per bogie for perfect balance
   - Create a specific redistribution plan showing how many passengers should move from which bogie to which
   - Provide a table showing current vs. ideal passenger distribution
   - Suggest a step-by-step implementation plan for station staff

2. **Passenger Flow Management**:
   - Design a specific boarding/alighting strategy for the next station
   - Recommend which doors should be prioritized for boarding vs. alighting
   - Suggest specific announcements that should be made to guide passengers
   - Provide a visual diagram or description of optimal passenger movement

3. **Real-time Adjustment Tactics**:
   - Recommend specific actions for train staff to balance crowds in real-time
   - Suggest targeted interventions for the most overcrowded bogies
   - Provide contingency plans if crowding worsens at upcoming stations
   - Recommend specific timing for implementing each redistribution action

4. **Bottleneck Resolution Strategy**:
   - Identify specific bottlenecks and congestion points within each bogie
   - Suggest precise methods to resolve each identified bottleneck
   - Recommend optimal standing positions to maximize space utilization
   - Provide guidance on luggage placement to optimize passenger flow

Format your response in markdown with clear headings, bullet points, and tables where appropriate. Provide specific, actionable insights rather than general observations."""
            
            payload = {
                "model": "llama3-70b-8192",  # Using the most capable model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized AI assistant for metro train crowd analysis with expertise in computer vision limitations, crowd counting, and passenger flow optimization. Your primary goal is to provide actionable recommendations for balancing passengers across train bogies and optimizing passenger comfort. Focus on creating specific, detailed redistribution plans that show exactly how many passengers should move from which bogie to which. Provide precise instructions for station staff on how to implement these plans. Include visual descriptions or diagrams when helpful. Your analysis should be highly specific, with exact numbers and percentages rather than general observations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Lower temperature for more consistent, factual responses
                "max_tokens": 2000,  # Allow for more detailed analysis
                "top_p": 0.95,  # Higher top_p for better quality
                "top_k": 40  # Add top_k parameter for better quality
            }
            
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                # Validate and enhance the analysis
                analysis = self.validate_groq_analysis(analysis, counts)
                return analysis
            else:
                # Use fallback analysis if API fails
                return self.generate_fallback_analysis(counts, recommendations, platform_data)
        
        except Exception as e:
            return self.generate_fallback_analysis(counts, recommendations, platform_data)
    
    def extract_groq_counts(self, analysis, default_counts):
        """Extract the estimated counts from Groq's analysis
        
        Args:
            analysis: Analysis text from Groq API
            default_counts: Default counts to use if extraction fails
            
        Returns:
            groq_counts: Extracted counts from Groq analysis
        """
        try:
            # Look for the table with estimated counts
            # Pattern to match the table rows with counts - updated for new format
            # | Bogie | Computer Vision Count | Your Estimated Count | Confidence (1-10) | Reasoning |
            pattern = r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
            
            # Find all matches
            matches = re.findall(pattern, analysis)
            
            if matches:
                # Extract the estimated counts (third column)
                groq_counts = [int(match[2]) for match in matches]
                print(f"Successfully extracted Groq counts: {groq_counts}")
                return groq_counts
            else:
                # Try alternative pattern with more flexible matching
                alt_pattern = r"\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|"
                alt_matches = re.findall(alt_pattern, analysis)
                
                if alt_matches:
                    # Sort by bogie number and extract counts
                    sorted_matches = sorted(alt_matches, key=lambda x: int(x[0]))
                    groq_counts = [int(match[1]) for match in sorted_matches]
                    print(f"Extracted Groq counts using alternative pattern: {groq_counts}")
                    return groq_counts
                
                # Try another alternative pattern for text-based descriptions
                text_pattern = r"Bogie\s+(\d+)[^0-9]*(\d+)\s+passengers"
                text_matches = re.findall(text_pattern, analysis, re.IGNORECASE)
                
                if text_matches:
                    # Sort by bogie number and extract counts
                    sorted_matches = sorted(text_matches, key=lambda x: int(x[0]))
                    groq_counts = [int(match[1]) for match in sorted_matches]
                    print(f"Extracted Groq counts using text pattern: {groq_counts}")
                    return groq_counts
                
                # Try one more pattern for estimated counts
                est_pattern = r"estimated\s+(?:actual\s+)?count\s+(?:for\s+)?(?:bogie\s+)?(\d+)[^0-9]*(\d+)"
                est_matches = re.findall(est_pattern, analysis, re.IGNORECASE)
                
                if est_matches:
                    # Create a dictionary to handle potentially out-of-order matches
                    count_dict = {}
                    for bogie_num, count in est_matches:
                        count_dict[int(bogie_num)] = int(count)
                    
                    # Convert to ordered list
                    groq_counts = [count_dict.get(i+1, default_counts[i] if i < len(default_counts) else 0) 
                                  for i in range(max(count_dict.keys()))]
                    print(f"Extracted Groq counts using estimated pattern: {groq_counts}")
                    return groq_counts
                
                # If no pattern matches, return the default counts
                print("No count pattern found in Groq analysis, using default counts")
                return default_counts
        except Exception as e:
            print(f"Error extracting Groq counts: {e}")
            return default_counts
    
    def validate_groq_analysis(self, analysis, counts):
        """Validate and enhance the Groq analysis to ensure quality
        
        Args:
            analysis: Analysis text from Groq API
            counts: List of people counts for each bogie
            
        Returns:
            enhanced_analysis: Validated and enhanced analysis
        """
        # Check if analysis contains required sections
        required_sections = [
            "passenger distribution", "redistribution plan", "bottlenecks", 
            "passenger flow", "adjustment tactics"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in analysis.lower():
                missing_sections.append(section)
        
        # If sections are missing, add placeholders
        if missing_sections:
            analysis += "\n\n## Additional Recommendations\n\n"
            
            # If redistribution plan is missing, add a more detailed one
            if "redistribution plan" in missing_sections:
                analysis += "### Optimal Passenger Redistribution Plan\n\n"
                
                # Calculate ideal distribution
                total_passengers = sum(counts)
                ideal_per_bogie = total_passengers / len(counts)
                
                # Create redistribution table
                analysis += "#### Current vs. Ideal Passenger Distribution\n\n"
                analysis += "| Bogie | Current Count | Ideal Count | Difference | Action |\n"
                analysis += "|-------|--------------|------------|------------|--------|\n"
                
                for i, count in enumerate(counts):
                    diff = count - ideal_per_bogie
                    action = "No action needed"
                    if diff > 5:
                        action = f"Redirect {int(diff)} passengers to less crowded bogies"
                    elif diff < -5:
                        action = f"Accept {int(abs(diff))} passengers from crowded bogies"
                    
                    analysis += f"| {i+1} | {count} | {int(ideal_per_bogie)} | {int(diff)} | {action} |\n"
                
                # Add specific movement instructions
                analysis += "\n#### Specific Passenger Movement Instructions\n\n"
                
                # Find overcrowded and undercrowded bogies
                overcrowded = [(i, count) for i, count in enumerate(counts) if count > ideal_per_bogie + 5]
                undercrowded = [(i, count) for i, count in enumerate(counts) if count < ideal_per_bogie - 5]
                
                if overcrowded and undercrowded:
                    for over_idx, over_count in overcrowded:
                        for under_idx, under_count in undercrowded:
                            move_count = min(over_count - int(ideal_per_bogie), int(ideal_per_bogie) - under_count)
                            if move_count > 0:
                                analysis += f"* Move {move_count} passengers from Bogie {over_idx+1} to Bogie {under_idx+1}\n"
                else:
                    analysis += "* The current distribution is relatively balanced. Minor adjustments can be made at the next station.\n"
            
            # Add other missing sections
            for section in missing_sections:
                if section != "redistribution plan":
                    analysis += f"### {section.title()}\n\n"
                    analysis += f"Based on the current passenger distribution, the following {section} is recommended:\n\n"
                    
                    if section == "passenger flow":
                        analysis += "* Direct new passengers to bogies with the most available capacity\n"
                        analysis += "* Prioritize alighting from the most crowded bogies first\n"
                        analysis += "* Make announcements to guide passengers to less crowded bogies\n"
                    elif section == "bottlenecks":
                        analysis += "* Monitor doorway areas in the most crowded bogies\n"
                        analysis += "* Ensure pathways remain clear for passenger movement\n"
                        analysis += "* Pay special attention to areas with luggage or standing passengers\n"
                    elif section == "adjustment tactics":
                        analysis += "* Station staff should position themselves near crowded bogies\n"
                        analysis += "* Use visual cues to direct passengers to less crowded areas\n"
                        analysis += "* Consider longer dwell times at busy stations to allow for redistribution\n"
        
        # Add factual data summary at the beginning
        summary = "## Summary of Current Crowd Distribution\n\n"
        summary += "| Bogie | Passenger Count | Density Level | Comfort Level |\n"
        summary += "|-------|-----------------|---------------|---------------|\n"
        
        for i, count in enumerate(counts):
            if count < 20:
                density = "Low"
                comfort = "Comfortable"
            elif count < 40:
                density = "Medium"
                comfort = "Moderately Crowded"
            else:
                density = "High"
                comfort = "Overcrowded"
                
            summary += f"| {i+1} | {count} | {density} | {comfort} |\n"
        
        # Add total and average statistics
        total_passengers = sum(counts)
        avg_per_bogie = total_passengers / len(counts) if counts else 0
        
        summary += f"\n**Total Passengers:** {total_passengers}  \n"
        summary += f"**Average Per Bogie:** {avg_per_bogie:.1f}  \n"
        
        # Calculate standard deviation to measure balance
        if len(counts) > 1:
            variance = sum((x - avg_per_bogie) ** 2 for x in counts) / len(counts)
            std_dev = variance ** 0.5
            balance_status = "Well Balanced" if std_dev < 5 else "Moderately Balanced" if std_dev < 10 else "Poorly Balanced"
            summary += f"**Balance Status:** {balance_status} (Standard Deviation: {std_dev:.1f})  \n"
        
        return summary + "\n\n" + analysis
    
    def generate_fallback_analysis(self, counts, recommendations, platform_data=None):
        """Generate a comprehensive analysis when the API fails
        
        Args:
            counts: List of people counts for each bogie
            recommendations: Recommendations for crowd redistribution
            platform_data: Platform data if available
            
        Returns:
            analysis: Detailed analysis text
        """
        # Create a comprehensive analysis based on the counts
        total = sum(counts)
        avg = total / len(counts) if counts else 0
        
        # Calculate standard deviation to measure balance
        if len(counts) > 1:
            variance = sum((x - avg) ** 2 for x in counts) / len(counts)
            std_dev = variance ** 0.5
            balance_status = "Well Balanced" if std_dev < 5 else "Moderately Balanced" if std_dev < 10 else "Poorly Balanced"
        else:
            std_dev = 0
            balance_status = "N/A"
        
        # Find most and least crowded bogies
        bogie_data = [(i+1, count) for i, count in enumerate(counts)]
        sorted_bogies = sorted(bogie_data, key=lambda x: x[1], reverse=True)
        
        # Calculate ideal distribution
        ideal_per_bogie = total / len(counts) if counts else 0
        
        # Create redistribution plan
        redistribution_plan = []
        overcrowded = [(i, count) for i, count in bogie_data if count > ideal_per_bogie + 5]
        undercrowded = [(i, count) for i, count in bogie_data if count < ideal_per_bogie - 5]
        
        for over_idx, over_count in overcrowded:
            for under_idx, under_count in undercrowded:
                move_count = min(over_count - int(ideal_per_bogie), int(ideal_per_bogie) - under_count)
                if move_count > 0:
                    redistribution_plan.append(f"Move {move_count} passengers from Bogie {over_idx} to Bogie {under_idx}")
        
        # Generate the analysis in markdown format
        analysis = """## Summary of Current Crowd Distribution

| Bogie | Passenger Count | Density Level | Comfort Level |
|-------|-----------------|---------------|---------------|
"""
        
        for i, count in enumerate(counts):
            if count < 20:
                density = "Low"
                comfort = "Comfortable"
            elif count < 40:
                density = "Medium"
                comfort = "Moderately Crowded"
            else:
                density = "High"
                comfort = "Overcrowded"
                
            analysis += f"| {i+1} | {count} | {density} | {comfort} |\n"
        
        analysis += f"""
**Total Passengers:** {total}  
**Average Per Bogie:** {avg:.1f}  
**Balance Status:** {balance_status} (Standard Deviation: {std_dev:.1f})  

## Part 1: Passenger Count Estimation

| Bogie | Computer Vision Count | Estimated Count | Confidence | Reasoning |
|-------|----------------------|----------------|------------|-----------|
"""
        
        for i, count in enumerate(counts):
            # Estimate slightly higher for crowded bogies to account for occlusions
            estimated = count
            if count > 20:
                estimated = int(count * 1.15)  # Add 15% for crowded bogies
                confidence = 7
                reasoning = "Likely undercounting due to occlusions in crowded areas"
            elif count > 10:
                estimated = int(count * 1.1)  # Add 10% for medium bogies
                confidence = 8
                reasoning = "Possible minor undercounting of seated passengers"
            else:
                estimated = count
                confidence = 9
                reasoning = "Accurate count due to low density"
                
            analysis += f"| {i+1} | {count} | {estimated} | {confidence}/10 | {reasoning} |\n"
        
        analysis += """
## Part 2: Detailed Crowd Analysis

### Computer Vision Accuracy Assessment
- The computer vision system appears to be performing well for this set of images
- Accuracy is likely highest in less crowded bogies (1-10 passengers)
- In more crowded bogies, there may be 10-15% undercounting due to occlusions
- Seated passengers near windows may be missed in some cases

### Passenger Distribution Analysis
"""
        
        analysis += f"""- Passengers are distributed unevenly across the train with a standard deviation of {std_dev:.1f}
- Most crowded bogie: Bogie {sorted_bogies[0][0]} with {sorted_bogies[0][1]} passengers
- Least crowded bogie: Bogie {sorted_bogies[-1][0]} with {sorted_bogies[-1][1]} passengers
- The current distribution is {balance_status.lower()}

### Comfort and Density Analysis
"""
        
        # Count bogies by comfort level
        comfortable = sum(1 for _, count in bogie_data if count < 20)
        moderate = sum(1 for _, count in bogie_data if 20 <= count < 40)
        overcrowded = sum(1 for _, count in bogie_data if count >= 40)
        
        analysis += f"""- {comfortable} bogies are in the "Comfortable" range (< 20 passengers)
- {moderate} bogies are "Moderately Crowded" (20-40 passengers)
- {overcrowded} bogies are "Overcrowded" (> 40 passengers)
- Standing density in the most crowded bogie is approximately {sorted_bogies[0][1]/5:.1f} passengers per square meter

## Part 3: Crowd Balancing and Redistribution Plan

### Optimal Passenger Distribution Plan

| Bogie | Current Count | Ideal Count | Difference | Action |
|-------|--------------|------------|------------|--------|
"""
        
        for i, count in enumerate(counts):
            diff = count - ideal_per_bogie
            action = "No action needed"
            if diff > 5:
                action = f"Redirect {int(diff)} passengers to less crowded bogies"
            elif diff < -5:
                action = f"Accept {int(abs(diff))} passengers from crowded bogies"
                
            analysis += f"| {i+1} | {count} | {int(ideal_per_bogie)} | {int(diff)} | {action} |\n"
        
        analysis += """
#### Specific Passenger Movement Instructions
"""
        
        if redistribution_plan:
            for plan in redistribution_plan:
                analysis += f"* {plan}\n"
        else:
            analysis += "* The current distribution is relatively balanced. Minor adjustments can be made at the next station.\n"
        
        analysis += """
### Passenger Flow Management
* Direct new passengers to bogies with the most available capacity
* Prioritize alighting from the most crowded bogies first
* Make announcements to guide passengers to less crowded bogies
* Station staff should position themselves near the doors of crowded bogies to assist with redistribution

### Real-time Adjustment Tactics
* Station staff should position themselves near crowded bogies
* Use visual cues to direct passengers to less crowded areas
* Consider longer dwell times at busy stations to allow for redistribution
* Implement a "wait for next train" policy if certain bogies reach maximum capacity

### Bottleneck Resolution Strategy
* Monitor doorway areas in the most crowded bogies
* Ensure pathways remain clear for passenger movement
* Pay special attention to areas with luggage or standing passengers
* Request passengers to move to the center of bogies to free up doorway areas
"""
        
        return analysis