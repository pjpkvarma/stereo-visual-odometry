# Stereo Visual Odometry

A real-time stereo visual odometry pipeline implemented using OpenCV in C++. This project estimates the camera trajectory by extracting ORB features, computing stereo disparity, reconstructing 3D points, and estimating pose using PnP with RANSAC. Visualizations include disparity maps and 2D trajectory plots.

[![Stereo Visual Odometry Demo](https://img.youtube.com/vi/nB3G3LbG-D0/hqdefault.jpg)](https://www.youtube.com/watch?v=nB3G3LbG-D0)

**Click the image above to watch the demo on YouTube**

## Key Features

- ORB Feature Detection & Tracking
- StereoSGBM-based Disparity Map
- Triangulation to get 3D points
- PnP Pose Estimation using RANSAC
- Real-time camera trajectory visualization
- Resized KITTI stereo dataset support with scaled intrinsics

## Technologies Used

- C++ & OpenCV
- KITTI Dataset (Grayscale Stereo)
- Visual Studio
- Disparity Estimation: StereoSGBM
- Pose Estimation: solvePnPRansac

## Output Examples

- Disparity Image  
- Keypoint Matches  
- Trajectory Plot  
