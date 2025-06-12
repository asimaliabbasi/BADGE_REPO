import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load your trained model
model = load_model('model/keras_model.h5')

# Load class labels
with open('model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Create white background
def create_white_bg():
    return np.full((480, 640, 3), 255, dtype=np.uint8)

# Camera setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create white background
    white_bg = create_white_bg()
    
    # Process frame for pose detection
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Draw pose on white background
        mp_drawing.draw_landmarks(
            white_bg,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
        )
        
        # Preprocess for model
        img = cv2.resize(white_bg, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id] * 100  # Convert to percentage
        label = class_names[class_id]
        
        # Add white background for text
        text = f"{label} {confidence:.0f}%"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 50), (255, 255, 255), -1)
        
        # Black text with percentage
        cv2.putText(frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Show what the model sees
        cv2.imshow("Model Input (White BG)", white_bg)
    
    # Main camera view
    cv2.imshow("Pose Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()