import cv2
import numpy as np

def run_part_2():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # --- Part 1 Recap: Segmentation ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # --- Part 2: Feature Extraction (New) ---
        # 1. Find the Contours (the edge of the hand)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Pick the largest contour (the hand)
            hand_contour = max(contours, key=cv2.contourArea)

            # 2. Draw the Convex Hull (The 'Rubber Band')
            hull = cv2.convexHull(hand_contour)
            cv2.drawContours(roi, [hull], -1, (255, 0, 0), 2) # Blue line

            # 3. Find Convexity Defects (The 'Valleys')
            hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
            defects = cv2.convexityDefects(hand_contour, hull_indices)

            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])

                    # Only count deep valleys (ignore noise)
                    if d > 10000:
                        finger_count += 1
                        cv2.circle(roi, far, 5, [0, 0, 255], -1) # Red dots for gaps

                # Display the count
                cv2.putText(frame, f"Fingers: {finger_count + 1}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Main Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_part_2()