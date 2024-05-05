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

# TILED MATRIX MULTIPLICATION 
Tiled matrix multiplication is implemented using Numba CUDA (numba tiled_mm.py) and C++ CUDA (c++ tiled_mm.cu). The Numba version is executed on RCHAU. For C++ CUDA version, the results are from the assignement (run on ASA-X) since RCHAU is currently inaccessible. But both the versions are executed on A100.

# Requirements for running Numba CUDA version on RCHAU
pip install numba

# Requirements for running C++ CUDA version on ASA-X 
The makefile with the name "Makefile" and the bash script named ""
