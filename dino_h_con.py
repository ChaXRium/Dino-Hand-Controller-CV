import cv2
import numpy as np

def build_hand_tracker():
    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Image Enhancement: Flip the frame (Mirror effect)
        frame = cv2.flip(frame, 1)
        
        # 2. Define a "Region of Interest" (ROI) 
        # This makes the code faster because it only looks at a small box
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # 3. Segmentation: Convert to HSV and filter for Skin Color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # These values work for most skin tones under normal light
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 4. Noise Reduction: Blur the mask to remove small dots
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Show the windows
        cv2.imshow("Main Feed", frame)
        cv2.imshow("Hand Mask", mask)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    build_hand_tracker()