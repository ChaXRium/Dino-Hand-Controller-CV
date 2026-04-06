import cv2
import os

# Setup folders
for folder in ['dataset/Jump', 'dataset/Idle']:
    if not os.path.exists(folder):
        os.makedirs(folder)

cap = cv2.VideoCapture(0)
print("Hold 'j' for Jump images | Hold 'i' for Idle images | 'q' to Finish")

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[100:350, 100:350] # Region of Interest
    
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)
    cv2.imshow("Capturing Data", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('j'):
        cv2.imwrite(f"dataset/Jump/img_{count}.jpg", roi)
        count += 1
    if key == ord('i'):
        cv2.imwrite(f"dataset/Idle/img_{count}.jpg", roi)
        count += 1
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} total images.")