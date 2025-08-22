Real-Time Eye Detection and Cropping with MediaPipe

This project demonstrates real-time eye detection, cropping, and visualization using MediaPipe Face Mesh and OpenCV.
It captures video from a webcam, identifies left and right eyes, draws bounding boxes on them, and displays cropped versions in separate windows.


Features
1) Real-time face mesh detection using MediaPipe
2) Accurate left and right eye detection (corrected for webcam mirror flip).
3) Automatic eye cropping with bounding boxes.
4) Cropped eyes displayed in separate resizable windows.
5)  Consistent crop size for both eyes (for clarity and ML preprocessing).
6)  Simple exit mechanism (q key).

Usage

Clone this repository or copy the script.
Run the Python script: main.py
>>
Three windows will appear:
Webcam with Eye Boxes → full webcam feed with bounding boxes.
Left Eye → cropped and resized left eye.
Right Eye → cropped and resized right eye.

Press q to exit.
