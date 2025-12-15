import cv2
import numpy as np

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open camera
cap = cv2.VideoCapture(0)

# Simple face ID counter (simulated recognition)
face_id = 0
known_faces = {}

print("Starting face recognition system...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Crop face region
        face_roi = gray[y:y+h, x:x+w]

        # Use face size as a simple feature (educational purpose)
        face_key = f"{w}_{h}"

        if face_key not in known_faces:
            face_id += 1
            known_faces[face_key] = face_id

        user_id = known_faces[face_key]

        # Draw rectangle
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)

        # Show ID on screen
        cv2.putText(frame,
                    f"User {user_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        # Simulate sending number to micro:bit
        print(f"Face recognized -> Display number {user_id} on Micro:bit")

    cv2.imshow("Face Recognition Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
