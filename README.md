# AI-Camera-DressCode
A real-time dress code detection system using YOLO. The model identifies and classifies clothing items (e.g., miniskirt, crop top) from camera feeds, detects violations, and highlights them with bounding boxes for monitoring and automated alerts.

## Overview
This project is an AI-powered camera system designed to detect and enforce dress code compliance for university students. It uses the YOLOv8n model, fine-tuned on a custom dataset to identify whether students are following the university's dress code based on their attire. The system processes real-time video or images from a camera, detects clothing items, and flags violations with bounding boxes and labels. Perfect for campus security or automated monitoring!
Key Features

Model: YOLOv8n, fine-tuned for high accuracy on dress code detection.
Dataset: Custom dataset of student attire images, annotated for specific dress code rules (e.g., formal vs. casual).
Functionality: Detects clothing items in real-time and classifies them as compliant or non-compliant.
Output: Visualizes results with bounding boxes and confidence scores.
Use Case: Automated dress code enforcement for universities.

## How It Works

Data Preparation: Collected and annotated images of students' outfits to train the model.
Fine-Tuning: Tuned YOLOv8n on the custom dataset to improve detection accuracy for specific clothing types.
Inference: The model processes input from a camera feed, identifies clothing, and checks compliance.
Visualization: Outputs results with bounding boxes around detected items and labels indicating compliance status.

## Setup and Usage

Clone this repo: git clone <repo-link>
Install dependencies: pip install -r requirements.txt
Run the Jupyter Notebook (Dress_Code_Detection.ipynb) for step-by-step training and inference.
Use inference.py for real-time camera detection.

Check the notebook for detailed code and results. Feel free to contribute or suggest improvements!
