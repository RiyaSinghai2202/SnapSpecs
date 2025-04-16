import cv2
import os
import numpy as np

# Load the webcam
cap = cv2.VideoCapture(0)

# Load the face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Overlay images folder and image counter
overlay_folder = r"C:\Users\Dell\OneDrive\Desktop\Virtual Try On\Glasses"  # Full path to folder
num = 1

# Function to update and load the overlay image
def update_overlay():
    global overlay_path, overlay
    overlay_path = os.path.join(overlay_folder, f'glass{num}.png')  # Match your filename pattern
    if not os.path.exists(overlay_path):
        print(f"[ERROR] Image not found: {overlay_path}")
        overlay = None
    else:
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            print(f"[ERROR] Failed to read image: {overlay_path}")

# Load the initial overlay image
update_overlay()

while True:
    key = cv2.waitKey(10)

    if key == ord('s'):
        num = (num % 29) + 1  # Cycle 1–29
        update_overlay()

    if key == ord('q'):
        break

    ret, frame = cap.read()
    if not ret or overlay is None:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        try:
            if len(eyes) >= 2:
                # Get two eyes sorted by x-position
                eyes = sorted(eyes, key=lambda e: e[0])
                eye1, eye2 = eyes[:2]

                # Compute centers of the eyes
                eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
                eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

                # Center point between eyes
                center_x = (eye1_center[0] + eye2_center[0]) // 2
                center_y = (eye1_center[1] + eye2_center[1]) // 2

                glasses_width = int(1.8 * abs(eye2_center[0] - eye1_center[0]))
                glasses_height = int(glasses_width * overlay.shape[0] / overlay.shape[1])

                x_pos = center_x - glasses_width // 2
                y_pos = center_y - glasses_height // 2

            else:
                # Fallback if eyes not detected
                glasses_width = int(w * 1.2)
                glasses_height = int(glasses_width * overlay.shape[0] / overlay.shape[1])
                x_pos = x + w//2 - glasses_width//2
                y_pos = y + h//3 - glasses_height//3

            # Keep inside frame
            if x_pos < 0 or y_pos < 0 or x_pos + glasses_width > frame.shape[1] or y_pos + glasses_height > frame.shape[0]:
                continue

            overlay_resized = cv2.resize(overlay, (glasses_width, glasses_height))

            alpha = overlay_resized[:, :, 3] / 255.0
            overlay_rgb = overlay_resized[:, :, :3]

            roi = frame[y_pos:y_pos+glasses_height, x_pos:x_pos+glasses_width]

            for c in range(0, 3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]

        except Exception as e:
            print(f"[ERROR] Overlay failed: {e}")
            continue

    cv2.imshow("Virtual Try-On: Glasses", frame)

cap.release()
cv2.destroyAllWindows()
