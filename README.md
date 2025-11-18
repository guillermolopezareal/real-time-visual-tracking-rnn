✨ Real-Time Visual Tracking Using Recurrent Regression Networks (RNN)

Fully Connected Networks (baseline detection)
Convolutional Neural Networks (CNNs)
Recurrent Neural Networks (RNN / ConvLSTM)
Regression layers predicting bounding box coordinates

The work is inspired by the paper:
📄 “Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects”.

🚀 Project Overview
Object tracking is a foundational task in computer vision, with applications in:

robotics
autonomous vehicles
video surveillance
gesture recognition
traffic monitoring

This project builds a custom dataset, a baseline detector, an RNN-based tracker, and evaluates everything using Intersection over Union (IoU) and loss metrics.

The final pipeline:
Object Detection Model (fully connected regression)
Object Tracking Model (ConvLSTM + dense layers)
Third-party baseline with OpenCV DNN for comparison
Synthetic Dataset Generator for controlled experiments

🧠 Methodology

1️⃣ Object Detection System
Fully connected neural network
Input: single frame
Output: bounding box → (x1, y1, x2, y2)
Loss: Mean Squared Error (MSE)

2️⃣ Object Tracking with ConvLSTM
Sequential input of frames
Convolutional layers extract appearance features
ConvLSTM layers model temporal dynamics
Dense layers regress bounding box
Evaluated with IoU

3️⃣ Dataset Generator
Custom Python tool that:
generates synthetic moving objects
outputs frames and labels
supports random and sequential motion
exports CSV annotation file

4️⃣ Benchmark using OpenCV DNN
Real-world demonstration using:
YOLO-like DNN
highway video
real-time car detection

📊 Key Results
  1. Object Detection
  2. Good bounding box regression
  3. IoU ≈ 0.5 → model finds ~50% overlap with ground truth
  4. Stable loss/accuracy curves

Object Tracking
  1. ConvLSTM architecture shows promising tendency
  2. Underperforms due to:
  3. limited dataset
  4. limited training time
  5. hardware constraints

OpenCV Baseline
  1. Strong real-world detection performance
  2. Serves as comparison point
  3. Detects multiple cars across frames

🧪 Sample Visualizations

![Dataset Generation](results/dataset_generator.png)

![IOU Metric](results/IOU.png)

![Training and Validation Loss](results/training_validation_loss.png)


🛠️ Technologies Used

Python
TensorFlow / Keras
NumPy / Pandas
Matplotlib
OpenCV
ConvLSTM layers
Custom synthetic dataset generator

📘 Full Report
📄 Read the full PDF report in this repository:
![Final Report](results/FinalReport.pdf)

