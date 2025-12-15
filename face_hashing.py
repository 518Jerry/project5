import cv2
import hashlib
import os
import time

def hash_face(face_image):
    """
    Convert face image into hashed value
    """
    # Resize face to fixed size
    face_resized = cv2.resize(face_image, (100, 100))

    # Convert image to bytes
    face_bytes = face_resized.tobytes()

    # Add salt for security
    salt = os.urandom(16)

    sha = hashlib.sha256()
    sha.update(salt + face_bytes)

    return sha.hexdigest(), salt


def save_hashed_face(hash_value, salt):
    """
    Save hashed face data (NOT raw image)
    """
    with open("hashed_faces.txt", "a") as f:
        f.write(f"{hash_value},{salt.hex()}\n")


# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Face data protection system started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        face_hash, face_salt = hash_face(face_img)
        save_hashed_face(face_hash, face_salt)

        print("Face captured and securely hashed")

        cv2.rectangle(frame, (x, y),
                      (x + w, y + h),
                      (255, 0, 0), 2)

        # Delay to avoid repeated storage
        time.sleep(2)

    cv2.imshow("Face Hashing Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
