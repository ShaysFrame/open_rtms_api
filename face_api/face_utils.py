import face_recognition
import numpy as np
from core.models import CustomUser


class FaceUtils:
    @staticmethod
    def process_image(image):
        # Convert image to numpy array
        img = face_recognition.load_image_file(image)
        face_encodings = face_recognition.face_encodings(img)

        detected_users = []
        for encoding in face_encodings:
            users = CustomUser.objects.all()
            for user in users:
                if user.face_encoding:
                    stored_encoding = np.frombuffer(
                        user.face_encoding, dtype=np.float64)
                    match = face_recognition.compare_faces(
                        [stored_encoding], encoding)
                    if match[0]:
                        detected_users.append(user.id)
        return detected_users
