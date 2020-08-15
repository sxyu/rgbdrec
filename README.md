# Overview
This is a common interface for registering point clouds captured using a depth camera (Kinect V2 or Azure Kinect), supporting COLMAP and SLAM-based solutions.
A general data capturing program is included.

## Binaries
- `rgbdrec`: Basic RGB-D sequence recorder 
- `rgbdreg-colmap`, `rgbdreg-orbslam2`: Point cloud registration (via odometry/SFM): supports COLMAP and ORB_SLAM2. Uses data format output by `rgbdrec`
    - `rgbdreg-colmap`: registration method based on COLMAP.
            Note the SFM is performed using color images only and is initially in an arbitrary unit;
            *we transform all matched feature points to camera space and compare the depth to
            the depth image to correct the scale*.
             Requires `colmap` to be in system PATH.
    - `rgbdreg-orbslam2`: registration method based on ORB_SLAM2. This method actually uses
        RGB-D information of each frame and produces correctly scaled camera poses.
- `rgbdreg-viewer`: Visualize point cloud registration. After running one of the above, use `rgbdreg-viewer <data-folder>` to ensure result is good.

![Screenshot of registered point cloud](https://github.com/sxyu/rgbdrec/blob/master/readme-img/registered.png?raw=true)

## Format
Each registration program produce a file `poses.txt` (readable with `numpy.loadtxt`) in the data directory. This is a text file of floats, separated by spaces and newlines, of shape [N, 12] where N is the number of images; each row can be viewed as a [3, 4] row-major rigid-body transform matrix (bottom row omitted) which transforms a point in homogeneous coordinates from **camera space to world space**. As usual, the 3 leftmost columns are the rotation and right column is the translation.

Regardless of program used, the output coordinate system will be right-handed 
with axis orientations `[x, y, z] = [right, up, backward (screen to your face)]`.

See `rgbdreg-viewer.cpp` for an example program reading poses.txt.

# Setup

## Dependencies
- OpenCV 3+
- Eigen 3
- Boost (really I only use program_options, filesystem, process)

## Optional Dependencies
- For recording, at least one of
    - libk4a (Azure Kinect SDK)
    - libfreenect2
- Registration software
    - For COLMAP just build and install https://github.com/colmap/colmap.
    - For ORB_SLAM2 the setup is more annoying. I use the binary ORB vocab format introduced by poine: https://github.com/poine/ORB_SLAM2. You can use my fork: https://github.com/sxyu/ORB_SLAM2 or one of many other versions to support OpenCV4 and make the build easier. Set 
    set `ORB_SLAM2_ROOT_DIR` environment variable to the repo root.
- Visualization
    - rgbdreg-viewer needs my OpenGL viewer https://github.com/sxyu/meshview. Follow instructions there to install.

## Build
Build using cmake as usual: `mkdir build && cd build && cmake .. && make -j12`

# Credit
Out of laziness I reuse a lot of code from my other project https://github.com/sxyu/avatar, which also borrows from https://github.com/augcog/OpenARK
