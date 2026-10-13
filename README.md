# Autonomous Vehicles Using YOLO Object Detection

## Overview
Autonomous vehicles need to understand their surroundings instantly and accurately. This project uses the YOLO (You Only Look Once) deep learning algorithm to detect and classify pedestrians, vehicles, cyclists, traffic signs and other obstacles in video in real time. Built with YOLOv3 and OpenCV's DNN module, it recognizes 80 object classes (COCO dataset) and draws labeled bounding boxes around each detection, making it a solid base for autonomous driving applications.

## Features
- Real-time detection with YOLOv3 + OpenCV DNN
- Configurable confidence threshold (default 0.6)
- Bounding boxes with class labels on video input

## Setup
```bash
git clone https://github.com/yourusername/autonomous-vehicles-yolo.git
cd autonomous-vehicles-yolo
pip install opencv-python numpy
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```
Place `yolov3.cfg` in the project folder and your input video as `video.mp4`.

## Usage
```bash
python a.py
```
Press `q` to quit.

## Performance
~30-45 FPS on GPU, suitable for real-time use.

## Future Work
Object tracking (SORT/DeepSORT), LiDAR/radar sensor fusion, edge-device optimization, 3D detection.

## References
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018)

## License
For educational and research use.
