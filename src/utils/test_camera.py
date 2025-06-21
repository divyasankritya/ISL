#!/usr/bin/env python3
"""
Simple script to test webcam access on macOS
"""

import cv2
import time
import sys

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(4):  # Try indices 0-3
        print(f"Attempting to access camera at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Could not open camera at index {camera_index}")
            continue
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Camera opened at index {camera_index} but couldn't read frames")
            cap.release()
            continue
            
        print(f"SUCCESS: Camera at index {camera_index} is working!")
        print(f"Frame shape: {frame.shape}")
        print("Displaying video feed. Press 'q' to quit.")
        
        # Show video feed for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 10:  # Show for 10 seconds
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame counter and instructions
            cv2.putText(frame, f"Camera {camera_index}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
            cv2.imshow(f"Camera Test (Index {camera_index})", frame)
            
            # Break loop if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    print("\nERROR: Could not access any camera.")
    print("Please check:")
    print("1. Camera permissions in System Preferences")
    print("2. If another application is using the camera")
    print("3. If your camera hardware is working properly")
    
    if sys.platform == 'darwin':  # macOS
        print("\nOn macOS:")
        print("- Go to System Preferences > Security & Privacy > Privacy > Camera")
        print("- Ensure Terminal or your Python IDE has camera access")
    
    return False

if __name__ == "__main__":
    test_camera() 