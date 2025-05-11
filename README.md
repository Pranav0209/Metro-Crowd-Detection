# 🚇 Metro Crowd Detection System

<p align="center">
  <img src="Demo/Metro_Coach_Color_change_GIF.gif" alt="Metro Crowd Indicator" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.7%2B-yellow" alt="Python Version">
</p>

## 📋 Overview

The **Metro Crowd Indicator System** is an AI-powered tool that analyzes and visualizes crowd density in metro train coaches. It helps passengers and metro operators by providing real-time information about occupancy across coaches.

## ❓ Why This Matters

Imagine if you could check the crowd status of each metro coach **before** the train even arrives:
- 🧍 Choose less crowded coaches
- 🕒 Wait strategically for the next train
- 🧭 Distribute passengers evenly
- 📊 Help stations manage flow efficiently

## ✨ Features

- **🔍 Person Detection**: YOLOv8 model detects passengers in each image
- **🎯 Density Classification**: Color-coded results (🔵 Blue = Low, 🟢 Green = Medium, 🔴 Red = High)
- **🚃 Multi-Coach Input**: Analyze up to 6 coaches at once
- **🧠 Smart Analysis**: Groq API provides detailed optimization suggestions
- **🌐 Web App**: Flask-based UI for easy image upload and result visualization

## 🛠️ Technical Architecture

- `YOLOv8` – detects people in metro bogie images  
- `Flask` – provides the web interface  
- `OpenCV` – processes and visualizes bounding boxes  
- `Groq API` – adds smart recommendations  

## 🔧 Installation

### ✅ Prerequisites
- Python 3.7+
- `pip` package manager

### 📦 Setup
```bash
git clone https://github.com/yourusername/metro-crowd-indicator.git
cd metro-crowd-indicator
pip install -r requirements.txt
```

## 🚀 Running the Application

1. **Start the Flask Server**
   ```bash
   python app.py
   ```

2. **Access the Web Interface**
   * Open your browser and go to: `http://127.0.0.1:5000`
   * Upload up to **6 images** of metro bogies
   * Click **"Analyze"** to get detection results and recommendations

## 📊 How It Works

### 🔄 Analysis Workflow
1. **Image Upload**: Through the Flask interface
2. **Detection**: YOLOv8 detects and counts people
3. **Crowd Classification**: Assigns a density color based on count
4. **Smart Suggestion Engine**: Groq API recommends redistributions
5. **Result Display**: Annotated images + summary overview

### 🖼️ Output Includes:
* Annotated images with bounding boxes
* A side-by-side summary of all bogies with color indicators
* Text recommendations from AI analysis

## ⚙️ Customization

### 🔧 Adjust Detection Sensitivity
In `process_metro_images()`:
```python
if model.names[cls] == "person" and conf > 0.3:
```
➡️ Increase or decrease `0.3` to adjust model sensitivity

### 🎯 Change Crowd Thresholds
In `create_summary_image()`:
```python
if count <= 5:
    color = (255, 0, 0)  # Blue - Low
elif count <= 10:
    color = (0, 255, 0)  # Green - Medium
else:
    color = (0, 0, 255)  # Red - High
```
➡️ Adjust thresholds and RGB color values to your preference
