import cv2
import numpy as np
import tensorflow as tf
import pyautogui

# 1. Load your Evolved Model
model = tf.keras.models.load_model('final_hand_model.h5')

cap = cv2.VideoCapture(0)
print("--- AI DINO CONTROLLER ACTIVE ---")
print("Open: https://chromedino.com/ and put your hand in the box.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # Use the same ROI (250x250) as training
    roi = frame[100:350, 100:350]
    
    # Preprocess (Resize to 64x64, Rescale is handled by the model layers)
    img = cv2.resize(roi, (64, 64))
    img = np.expand_dims(img, axis=0) # Add batch dimension
    
    # 2. Prediction
    prediction = model.predict(img, verbose=0)[0][0]
    
    # Binary Threshold (0.3)
    if prediction > 0.3:
        label = "JUMP"
        color = (0, 255, 0)
        pyautogui.press('space') # Trigger the jump!
    else:
        label = "IDLE"
        color = (0, 0, 255)

    # 3. Visual UI
    cv2.putText(frame, f"STATUS: {label} ({prediction:.2f})", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 2)
    
    cv2.imshow("Dino AI Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()