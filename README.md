



Real-Time Hand Proximity Detection System

------------------------------------------



A Computer Vision Prototype Using Classical CV Techniques



This project implements a real-time hand-tracking prototype that detects how close a user’s hand is to a virtual on-screen boundary and classifies the interaction into three states:



SAFE – Hand is far from the boundary



WARNING – Hand is approaching



DANGER – Hand is extremely close / touching



Displays an on-screen alert: “DANGER DANGER”



The system is built using Flask + OpenCV and avoids all pose-detection libraries such as MediaPipe or OpenPose.

Only classical computer-vision techniques are used (color segmentation, thresholding, contours, convex hull, etc.).



&nbsp; Features

-----------



Real-time webcam-based hand tracking



Virtual boundary region drawn on the frame



Distance-based state logic



Dynamic visual feedback overlay



No pose-detection APIs (✔ requirement)



Achieves 8+ FPS on CPU-only execution



Simple and lightweight architecture



Works fully in the browser via Flask server

