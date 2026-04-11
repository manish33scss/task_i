# Aerial Guardian - Drone-based Person Detection & Tracking

## Overview

This project implements a lightweight person detection and tracking pipeline for drone aerial footage. The system handles small object detection, camera motion, and varying lighting conditions.

**Target Class:** Persons (pedestrians/people)

## Architecture

Input Video → Gamma Correction → YOLOv8n Detection → PineSORT Tracking → Output Video


## Components

### 1. Detection - YOLOv8n

- **Base Model:** YOLOv8n (nano) - lightweight, <10MB
- **Training Strategy:** 
  - Started with pretrained weights from **CrowdHuman** dataset (excellent for crowded person detection)
  - Fine-tuned on **VisDrone Detection** dataset for drone-specific aerial views
  - Merged `pedestrian` (class 1) and `people` (class 2) into single `human` class

**Training Results:**
| Metric | Value |
|--------|-------|
| mAP50 | 31.1% |
| Precision | 75% |
| Recall | 70% |
| Model Size | ~6 MB |

> *Training plots and confusion matrix are available in the `/training_results` folder*

### 2. Tracking - PineSORT

**Why PineSORT instead of ByteTrack?**
- ByteTrack suffered from **constant ID switching** due to drone camera movement
- PineSORT's camera compensation mechanism significantly reduced ID switches
- Better handles ego-motion of the drone platform

**Key Features Used:**
- Camera compensation with ORB features
- Kalman filter with motion prediction
- IoU-based association with multiple thresholds

### 3. Preprocessing - Gamma Correction

- Improves visibility in dark or unevenly lit scenes
- Adaptive gamma value based on image brightness
- Particularly helpful for dawn/dusk drone footage

## Dataset

- **Training:** VisDrone Detection Dataset (6,471 images)
- **Evaluation:** VisDrone MOT Validation Set (video sequences with ground truth)
- **Classes:** Merged pedestrian + people → human

## Usage

### Installation

# Clone repository
git clone https://github.com/manish33scss/task_i.git
cd task_i

# Install dependencies
pip install -r requirements.txt

# System Configuration
Training : i5 9th gen, nvidia 1650 gtx - Laptop
Testing : same laptop, jetson nano
IDE : Spyder

# Execution
- After cloning and installing necessary packages.
- I have created files 1) Bytetracker 2) pinesortTracker
- Both of them are executable by simply using

python3 byteTracker.py
python3 pinesortTracker.py

PS : you will have to change path in the file itself, both of the scripts are commented enough to give you understanding to where the change the file path.  

#Jetson Nano
For jetson nano, i used docker version of yolov8.
- here i converted .pt file to tensorrt file (.engine) for better performance, on video - with tracker i am getting 45ms per frame. 
expected performance : 20-30 FPS with FP16 - 640 resolution.


