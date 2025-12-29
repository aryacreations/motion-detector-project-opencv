import cv2
import numpy as np
import time

class MotionDetector:
    def __init__(self, min_area=500, learning_rate=0.01):
        """
        Initialize Motion Detector
        
        Args:
            min_area: Minimum contour area to be considered as motion
            learning_rate: Background subtractor learning rate (0-1)
        """
        self.min_area = min_area
        self.learning_rate = learning_rate
        
        # Create background subtractor using MOG2 algorithm
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # For FPS calculation
        self.prev_time = 0
        
    def detect_motion(self, frame):
        """
        Detect motion in the given frame
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with bounding boxes around moving objects
            motion_count: Number of moving objects detected
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows (set shadow pixels to 0)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce noise
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        # Threshold to get binary image
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around detected motion
        motion_count = 0
        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle around moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            
            # Add label
            cv2.putText(frame, f"Motion {motion_count + 1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            motion_count += 1
        
        return frame, motion_count, fg_mask
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        return fps
    
    def reset_background(self):
        """Reset the background model"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )


def main():
    """Main function to run motion detection"""
    print("=" * 60)
    print("Motion Detection Application")
    print("=" * 60)
    print("\nControls:")
    print("  'q' - Quit application")
    print("  'r' - Reset background model")
    print("  's' - Toggle mask view")
    print("  '+' - Increase sensitivity (decrease min area)")
    print("  '-' - Decrease sensitivity (increase min area)")
    print("\n" + "=" * 60)
    
    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize motion detector
    detector = MotionDetector(min_area=500)
    
    # Flag to toggle mask view
    show_mask = False
    
    print("\nStarting motion detection... Please wait for camera to initialize.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect motion
        processed_frame, motion_count, fg_mask = detector.detect_motion(frame)
        
        # Calculate FPS
        fps = detector.calculate_fps()
        
        # Add information overlay
        info_text = [
            f"FPS: {fps:.1f}",
            f"Moving Objects: {motion_count}",
            f"Min Area: {detector.min_area}",
            f"Press 'q' to quit"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display the frame
        if show_mask:
            # Convert mask to 3-channel for display
            fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((processed_frame, fg_mask_colored))
            cv2.imshow('Motion Detection (Original | Mask)', combined)
        else:
            cv2.imshow('Motion Detection', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting application...")
            break
        elif key == ord('r'):
            print("Resetting background model...")
            detector.reset_background()
        elif key == ord('s'):
            show_mask = not show_mask
            print(f"Mask view: {'ON' if show_mask else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            detector.min_area = max(100, detector.min_area - 100)
            print(f"Sensitivity increased (min area: {detector.min_area})")
        elif key == ord('-') or key == ord('_'):
            detector.min_area = min(5000, detector.min_area + 100)
            print(f"Sensitivity decreased (min area: {detector.min_area})")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")


if __name__ == "__main__":
    main()
