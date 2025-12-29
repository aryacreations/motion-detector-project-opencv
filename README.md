# Motion Detection Application 🎥

A real-time motion detection system that uses OpenCV to detect and highlight moving objects from your webcam feed.

## Features ✨

- **Real-time Motion Detection**: Detects moving objects using advanced background subtraction
- **Visual Feedback**: Draws bounding boxes and contours around detected motion
- **Configurable Sensitivity**: Adjust detection threshold on-the-fly
- **Performance Monitoring**: Real-time FPS counter
- **Mask Visualization**: Toggle between normal view and mask view
- **Background Reset**: Reset the background model when needed

## How It Works 🔧

The application uses the **MOG2 (Mixture of Gaussians)** background subtraction algorithm:

1. **Background Modeling**: Builds a statistical model of the background
2. **Foreground Detection**: Identifies pixels that differ from the background
3. **Noise Reduction**: Applies morphological operations and Gaussian blur
4. **Contour Detection**: Finds contours of moving objects
5. **Visualization**: Draws bounding boxes around detected motion

## Installation 📦

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀

Run the motion detector:

```bash
python motion_detector.py
```

### Keyboard Controls ⌨️

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `r` | Reset background model |
| `s` | Toggle mask view (show/hide foreground mask) |
| `+` or `=` | Increase sensitivity (decrease minimum area) |
| `-` or `_` | Decrease sensitivity (increase minimum area) |

## Configuration ⚙️

You can modify the `MotionDetector` class initialization in the code:

```python
detector = MotionDetector(
    min_area=500,        # Minimum contour area (pixels)
    learning_rate=0.01   # Background learning rate (0-1)
)
```

### Parameters:

- **min_area**: Minimum contour area to be considered as motion (default: 500)
  - Lower values = more sensitive (detects smaller movements)
  - Higher values = less sensitive (only detects larger movements)

- **learning_rate**: How quickly the background model adapts (default: 0.01)
  - Lower values = slower adaptation (better for static scenes)
  - Higher values = faster adaptation (better for changing environments)

## Troubleshooting 🔍

### Camera Not Opening
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Too Many False Detections
- Increase the `min_area` parameter (press `-` key while running)
- Reduce the `learning_rate` for more stable background modeling

### Motion Not Detected
- Decrease the `min_area` parameter (press `+` key while running)
- Press `r` to reset the background model
- Ensure there's adequate lighting

### Performance Issues
- Reduce camera resolution in the code:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```

## Technical Details 📊

### Algorithm: MOG2 Background Subtraction

- **History**: 500 frames for background modeling
- **Variance Threshold**: 16 (sensitivity to pixel changes)
- **Shadow Detection**: Enabled (shadows are removed from detection)

### Image Processing Pipeline:

1. Background subtraction
2. Shadow removal
3. Morphological closing (fill gaps)
4. Morphological opening (remove noise)
5. Gaussian blur (smooth edges)
6. Binary thresholding
7. Contour detection
8. Bounding box drawing

## Use Cases 💡

- **Security Monitoring**: Detect intruders or unexpected movement
- **Wildlife Observation**: Track animal movement
- **Traffic Analysis**: Monitor vehicle or pedestrian traffic
- **Smart Home**: Trigger actions based on motion detection
- **Computer Vision Learning**: Understand motion detection algorithms

## Requirements 📋

- Python 3.7+
- OpenCV 4.8.0+
- NumPy 1.24.0+
- Webcam or video input device

## License 📄

This project is open source and available for educational and personal use.

## Contributing 🤝

Feel free to fork this project and submit pull requests for improvements!

---

**Enjoy detecting motion!** 🎯
