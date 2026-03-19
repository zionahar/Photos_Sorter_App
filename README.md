# Photos Sorter App

A Python application for sorting photos by detected faces using advanced computer vision techniques.

## Overview

This project uses MediaPipe for face detection, InceptionResnetV1 for face embeddings, and clustering algorithms to group and classify faces in photo collections. It annotates images with bounding boxes and class labels, and generates visualization plots.

## Features

- Face detection and alignment using MediaPipe FaceMesh
- Face embedding extraction with InceptionResnetV1
- Automatic clustering with silhouette-guided KMeans
- Outlier detection for non-face images
- 2D PCA scatter plot visualization
- Annotated output images with bounding boxes and labels

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zionahar/Photos_Sorter_App.git
   cd Photos_Sorter_App
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your photos in the `data/` directory.

2. Run the face classifier:
   ```bash
   python face_classifier.py
   ```

3. View results in the `output/` directory:
   - Annotated images with bounding boxes and class labels
   - `cluster_scatter.png` for clustering visualization

## Dependencies

- opencv-python
- numpy
- torch
- facenet-pytorch
- mediapipe
- scikit-learn
- matplotlib

## Project Structure

- `face_classifier.py`: Main script for face classification and clustering
- `face_detection.py`: Face detection utilities
- `mediapipe_iris.py`: MediaPipe iris detection (if used)
- `requirements.txt`: Python dependencies
- `ckpt/`: Pretrained model checkpoints
- `data/`: Input photos directory
- `output/`: Processed results directory

## Contributing

Feel free to submit issues and pull requests.

## License

This project is private and for personal use.