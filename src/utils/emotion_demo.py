import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that doesn't require a GUI framework
import matplotlib.pyplot as plt

# Define emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_emotion_model(model_path):
    """
    Load the pre-trained emotion detection model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading emotion model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    return model

def emotion_analysis(emotions, output_path='emotion_analysis.png'):
    """
    Create a bar chart visualization of emotion predictions and save to file
    """
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(10, 6))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10, width=4, length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    # Save the plot to a file instead of displaying it
    plt.savefig(output_path)
    plt.close()
    
    print(f"Emotion analysis saved to {output_path}")

def preprocess_image(image, face_cascade=None):
    """
    Preprocess the image for emotion detection:
    1. Convert to grayscale if colored
    2. Detect face if cascade classifier is provided
    3. Resize to 48x48
    4. Normalize pixel values
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect face if face cascade is provided
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Take the largest face
            max_area = 0
            max_face = None
            for (x, y, w, h) in faces:
                if w*h > max_area:
                    max_area = w*h
                    max_face = (x, y, w, h)
            
            if max_face is not None:
                x, y, w, h = max_face
                gray = gray[y:y+h, x:x+w]
    
    # Resize to 48x48 pixels (model input size)
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Reshape for model input
    preprocessed = normalized.reshape(1, 48, 48, 1)
    
    return preprocessed, gray

def predict_emotion(model, image):
    """
    Predict emotion from an image
    """
    # Load the haar cascade for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        print("Warning: Haar cascade file not found. Face detection will be skipped.")
        face_cascade = None
    
    # Initialize face coordinates
    face_coords = None
    
    # Convert to grayscale for face detection
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect face if face cascade is provided
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Take the largest face
            max_area = 0
            max_face = None
            for (x, y, w, h) in faces:
                if w*h > max_area:
                    max_area = w*h
                    max_face = (x, y, w, h)
            
            if max_face is not None:
                x, y, w, h = max_face
                face_coords = (x, y, w, h)
                face_roi = gray[y:y+h, x:x+w]
                # Resize for model input
                face_roi = cv2.resize(face_roi, (48, 48))
                # Normalize pixel values
                normalized = face_roi / 255.0
                # Reshape for model input
                preprocessed = normalized.reshape(1, 48, 48, 1)
    else:
        # If no face detected, just resize the whole image
        preprocessed, _ = preprocess_image(image)
    
    # Predict emotion
    predictions = model.predict(preprocessed)[0]
    
    # Get the emotion label and probability
    emotion_idx = np.argmax(predictions)
    emotion = EMOTIONS[emotion_idx]
    probability = predictions[emotion_idx]
    
    # Get all emotion probabilities in a dictionary
    all_probabilities = {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))}
    
    return emotion, probability, all_probabilities, gray, face_coords

def display_results(image, emotion, probability, all_probabilities, output_path='emotion_results.png'):
    """
    Display the original image and emotion prediction results by saving to a file
    """
    plt.figure(figsize=(12, 6))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    # Display emotion prediction
    plt.subplot(1, 2, 2)
    
    # Display the main detected emotion
    plt.text(0.5, 0.9, f"Detected Emotion: {emotion}", 
             horizontalalignment='center', fontsize=16, weight='bold')
    plt.text(0.5, 0.8, f"Confidence: {probability*100:.1f}%", 
             horizontalalignment='center', fontsize=14)
    
    # Create bar chart of all emotions
    emotions = list(all_probabilities.keys())
    probs = [all_probabilities[e] for e in emotions]
    
    plt.barh(range(len(emotions)), probs, align='center')
    plt.yticks(range(len(emotions)), emotions)
    plt.xlim(0, 1)
    plt.xlabel('Probability')
    plt.title('Emotion Probabilities')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Results visualization saved to {output_path}")

def detect_from_file(model, image_path):
    """
    Detect emotion from an image file
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}.")
        return
    
    # Predict emotion
    emotion, probability, all_probabilities, processed_image, face_coords = predict_emotion(model, image)
    
    # Print results
    print(f"Detected Emotion: {emotion}")
    print(f"Confidence: {probability*100:.1f}%")
    
    # Draw face rectangle on the image if face was detected
    if face_coords is not None:
        x, y, w, h = face_coords
        # Get color based on emotion
        if emotion == 'Happy':
            color = (0, 255, 0)  # Green for happy
        elif emotion == 'Angry' or emotion == 'Fear':
            color = (0, 0, 255)  # Red for negative emotions
        elif emotion == 'Surprise':
            color = (255, 255, 0)  # Cyan for surprise
        else:
            color = (255, 255, 255)  # White for others
        
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    # Display results
    display_results(image, emotion, probability, all_probabilities)
    
    # Add standalone emotion analysis visualization
    emotion_values = [all_probabilities[e] for e in EMOTIONS]
    emotion_analysis(emotion_values)
    
    return emotion, probability

def detect_from_webcam(model):
    """
    Detect emotions in real-time from webcam
    """
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit, 'c' to capture and analyze current frame.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display frame
        cv2.imshow('Webcam (Press q to quit, c to capture)', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Quit if 'q' is pressed
        if key == ord('q'):
            break
        
        # Analyze current frame if 'c' is pressed
        elif key == ord('c'):
            # Predict emotion
            emotion, probability, all_probabilities, processed_image, face_coords = predict_emotion(model, frame)
            
            # Print results
            print(f"Detected Emotion: {emotion}")
            print(f"Confidence: {probability*100:.1f}%")
            
            # Display results
            display_results(frame, emotion, probability, all_probabilities)
            
            # Add standalone emotion analysis visualization
            emotion_values = [all_probabilities[e] for e in EMOTIONS]
            emotion_analysis(emotion_values)
    
    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emotion detection demo')
    parser.add_argument('--model', type=str, default='models/emotion_models.h5',
                        help='Path to pre-trained emotion model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image file (if not using webcam)')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for real-time emotion detection')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only show the emotion analysis chart for predictions')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    results_path = os.path.join(args.output_dir, 'emotion_results.png')
    analysis_path = os.path.join(args.output_dir, 'emotion_analysis.png')
    
    # Load emotion model
    model = load_emotion_model(args.model)
    
    # Detect emotions
    if args.webcam:
        print("Running webcam mode. This will save captures to image files.")
        print("Displaying emotion detection in real-time on the webcam feed.")
        
        # Try different camera indices
        camera_index = 0
        max_attempts = 3
        cap = None
        
        for attempt in range(max_attempts):
            print(f"Trying to access camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened() and cap.read()[0]:
                print(f"Successfully connected to camera at index {camera_index}")
                break
            else:
                print(f"Failed to access camera at index {camera_index}")
                if cap.isOpened():
                    cap.release()
                camera_index += 1
                
        if cap is None or not cap.isOpened():
            print("Error: Could not open any webcam. Please check your camera permissions in System Preferences.")
            print("If using macOS, make sure to grant camera access to Terminal or your Python application.")
            return
        
        # Try to read a test frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("Error: Camera opened but couldn't capture frames.")
            print("Please check if another application is using your camera.")
            cap.release()
            return
            
        print("Camera working correctly!")
        print("Press 'c' to capture and analyze current frame, 'q' to quit.")
        
        frame_count = 0
        
        # For performance, don't run detection on every single frame
        process_every_n_frames = 10
        frame_index = 0
        current_emotion = None
        current_probability = None
        current_face_coords = None
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Process frame for emotion detection every N frames
            frame_index += 1
            if frame_index % process_every_n_frames == 0:
                # Predict emotion
                emotion, probability, all_probabilities, _, face_coords = predict_emotion(model, frame)
                current_emotion = emotion
                current_probability = probability
                current_face_coords = face_coords
            
            # Display the detected emotion on the frame
            display_frame = frame.copy()
            
            # Draw rectangle around face if detected
            if current_face_coords is not None:
                x, y, w, h = current_face_coords
                # Get color based on emotion
                if current_emotion == 'Happy':
                    color = (0, 255, 0)  # Green for happy
                elif current_emotion == 'Angry' or current_emotion == 'Fear':
                    color = (0, 0, 255)  # Red for negative emotions
                elif current_emotion == 'Surprise':
                    color = (255, 255, 0)  # Cyan for surprise
                else:
                    color = (255, 255, 255)  # White for others
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Add emotion label above the rectangle
                label_y = max(y - 10, 0)  # Ensure label is visible
                cv2.putText(display_frame, current_emotion, (x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add a semi-transparent overlay at the bottom of the frame
            if current_emotion is not None:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, display_frame.shape[0] - 40), 
                             (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                
                # Add text with emotion and confidence
                text = f"{current_emotion}: {current_probability*100:.1f}%"
                cv2.putText(display_frame, text, (10, display_frame.shape[0] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection (Press q to quit, c to capture)', display_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Quit if 'q' is pressed
            if key == ord('q'):
                break
            
            # Analyze current frame if 'c' is pressed
            elif key == ord('c'):
                frame_count += 1
                
                # Save the captured frame
                capture_path = os.path.join(args.output_dir, f'capture_{frame_count}.jpg')
                cv2.imwrite(capture_path, frame)
                print(f"Frame captured and saved to {capture_path}")
                
                # Make sure we have the latest emotion prediction for this frame
                emotion, probability, all_probabilities, processed_image, face_coords = predict_emotion(model, frame)
                current_emotion = emotion
                current_probability = probability
                current_face_coords = face_coords
                
                # Print results
                print(f"Detected Emotion: {emotion}")
                print(f"Confidence: {probability*100:.1f}%")
                
                # Save results visualizations
                frame_results_path = os.path.join(args.output_dir, f'emotion_results_{frame_count}.png')
                frame_analysis_path = os.path.join(args.output_dir, f'emotion_analysis_{frame_count}.png')
                
                display_results(frame, emotion, probability, all_probabilities, frame_results_path)
                
                # Add standalone emotion analysis visualization
                emotion_values = [all_probabilities[e] for e in EMOTIONS]
                emotion_analysis(emotion_values, frame_analysis_path)
        
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        
    elif args.image:
        # Use provided image file
        if not os.path.exists(args.image):
            print(f"Error: File {args.image} not found.")
            return
            
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Unable to read image {args.image}.")
            return
            
        # Predict emotion
        emotion, probability, all_probabilities, processed_image, face_coords = predict_emotion(model, image)
        
        if args.analyze_only:
            # Only show the emotion analysis chart
            emotion_values = [all_probabilities[e] for e in EMOTIONS]
            emotion_analysis(emotion_values, analysis_path)
        else:
            # Save full results
            print(f"Detected Emotion: {emotion}")
            print(f"Confidence: {probability*100:.1f}%")
            display_results(image, emotion, probability, all_probabilities, results_path)
            
            # Also save the standalone emotion analysis
            emotion_values = [all_probabilities[e] for e in EMOTIONS]
            emotion_analysis(emotion_values, analysis_path)
    else:
        # No input specified, show help
        parser.print_help()
        print("\nError: Please specify either --image or --webcam.")

if __name__ == "__main__":
    main() 