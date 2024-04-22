# Object-Tracking
This is a video processing system for object tracking using Python. Simple Online and Realtime Tracking (SORT) algorithm has been used to
implement multi-object tracking in video streams. SORT algorithm combines the principles of the Kalman filter for state prediction and the IoU for data association.The Kalman filter is a recursive mathematical algorithm that estimates the state of a linear dynamic system from a series of noisy measurements over time. IoU is used to associate detections with existing tracks. To parallelize this object-tracking on GPU, Numba library has been used with its CUDA JIT compiler for writing CUDA kernels in Python.

# Dependencies
python 3.11.7

YOLOV3 pre-trained models

# Requirements
python -m pip install -r requirements.txt

## yolov3.weights [Download here: https://drive.google.com/file/d/1iFK2-I6SI23A6wTnGOTx7kMfcM7OsVlT/view?usp=sharing] 

# Execution 
A demo video (demo.mp4) has been provided to test the code. An output video (output.mp4) with the objects detected has also been provided.
