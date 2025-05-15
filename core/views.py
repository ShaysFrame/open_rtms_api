import cv2
import numpy as np
import torch
from ultralytics import YOLO
import face_recognition  # Still useful for face embeddings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Student, AttendanceRecord
from .serializers import StudentSerializer
import time

# Load the YOLOv8 face detection model once at module level
# or 'yolov8s-face.pt' for better accuracy
face_detector = YOLO(
    '/Users/shay/Dev/Projects/project_open_rtms/open_rtms_api/yolov8n-face-lindevs.pt')


class RegisterFaceView(APIView):
    def post(self, request):
        name = request.data.get('name')
        student_id = request.data.get('student_id')
        photo = request.FILES.get('photo')

        if not photo or not name or not student_id:
            return Response({'error': 'Missing fields'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Load the image using OpenCV
            img_data = photo.read()
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return Response({'error': 'Invalid image format'}, status=status.HTTP_400_BAD_REQUEST)

            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces using YOLOv8
            results = face_detector(rgb_image)

            # Extract the detected face with the highest confidence
            if len(results[0].boxes) == 0:
                return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

            # Get bounding boxes in (x1, y1, x2, y2) format
            boxes = results[0].boxes.xyxy.cpu().numpy()
            # Get confidence scores
            confidences = results[0].boxes.conf.cpu().numpy()

            # Find the face with highest confidence
            best_face_idx = np.argmax(confidences)
            x1, y1, x2, y2 = map(int, boxes[best_face_idx])

            # Extract the face region
            face_img = rgb_image[y1:y2, x1:x2]

            # Generate face embedding using face_recognition with robust error handling
            # This step generates a 128-dimension face encoding vector
            try:
                # Ensure face image has good dimensions
                min_face_size = 150
                if face_img.shape[0] < min_face_size or face_img.shape[1] < min_face_size:
                    # Calculate new size maintaining aspect ratio
                    scale = max(
                        min_face_size / face_img.shape[0], min_face_size / face_img.shape[1])
                    new_size = (
                        int(face_img.shape[1] * scale), int(face_img.shape[0] * scale))
                    face_img = cv2.resize(
                        face_img, new_size, interpolation=cv2.INTER_CUBIC)
                    print(f"Resized registration face to {face_img.shape}")

                # Try direct encoding
                encodings = face_recognition.face_encodings(
                    face_img, num_jitters=3)
            except Exception as e:
                print(f"Direct encoding failed: {str(e)}")
                encodings = []

            if not encodings:
                try:
                    # Try with face locations first
                    face_locations = face_recognition.face_locations(face_img)
                    if face_locations:
                        encodings = face_recognition.face_encodings(
                            face_img, face_locations, num_jitters=3)
                except Exception as e:
                    print(f"Encoding with face_locations failed: {str(e)}")
                    encodings = []

            if not encodings:
                try:
                    # Try with the whole image if face crop fails
                    # Convert to face_recognition format
                    face_location = [(y1, x2, y2, x1)]
                    encodings = face_recognition.face_encodings(
                        rgb_image, face_location, num_jitters=3)
                except Exception as e:
                    print(f"Encoding with original image failed: {str(e)}")

            if not encodings:
                return Response({'error': 'Could not generate face encoding. Please try again with a clearer photo.'},
                                status=status.HTTP_400_BAD_REQUEST)

            embedding = encodings[0].tobytes()

            # Create student record in database
            student = Student.objects.create(
                name=name,
                student_id=student_id,
                photo=photo,
                embedding=embedding
            )

            return Response(StudentSerializer(student).data, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"Error during registration: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RecognizeFaceView(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        recognized_by = request.data.get('recognized_by', 'mobile_app')
        session_id = request.data.get('session_id', 'unknown_session')
        print(f"Session ID: {session_id}")
        print(f"Got recognition request with recognized_by: {recognized_by}")

        if not image_file:
            print("ERROR: No image file found in request")
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        print(
            f"Image file received: {image_file.name}, size: {image_file.size} bytes")

        try:
            # Load the image
            img_data = image_file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                print("ERROR: Could not decode image")
                return Response({'error': 'Invalid image format'}, status=status.HTTP_400_BAD_REQUEST)

            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded, shape: {rgb_image.shape}")

            # Save original image for debugging (optional)
            timestamp = int(time.time())
            debug_path = f"/tmp/original_{timestamp}.jpg"
            cv2.imwrite(debug_path, image)
            print(f"Saved debug image to {debug_path}")

            # Detect faces using YOLOv8
            results = face_detector(rgb_image)

            if len(results[0].boxes) == 0:
                print("ERROR: No faces detected by YOLOv8")
                return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            print(f"YOLOv8 detected {len(boxes)} faces")

            # Process each detected face
            faces_data = []
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                if conf < 0.5:  # Skip low confidence detections
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Extract face region
                face_img = rgb_image[y1:y2, x1:x2]

                # Save detected face for debugging
                face_debug_path = f"/tmp/face_{i}_{timestamp}.jpg"
                cv2.imwrite(face_debug_path, cv2.cvtColor(
                    face_img, cv2.COLOR_RGB2BGR))

                try:
                    # Generate face embedding using a safer approach
                    # Ensure image has correct dimensions and type
                    # face_img must be RGB and uint8
                    if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                        # First, ensure the image is resized to a reasonable size
                        # Too small faces can cause problems with dlib
                        min_face_size = 150
                        if face_img.shape[0] < min_face_size or face_img.shape[1] < min_face_size:
                            # Calculate new size maintaining aspect ratio
                            scale = max(
                                min_face_size / face_img.shape[0], min_face_size / face_img.shape[1])
                            new_size = (
                                int(face_img.shape[1] * scale), int(face_img.shape[0] * scale))
                            face_img = cv2.resize(
                                face_img, new_size, interpolation=cv2.INTER_CUBIC)
                            print(f"Resized face to {face_img.shape}")

                        # Method 1: Try direct encoding
                        try:
                            encoding = face_recognition.face_encodings(
                                face_img, num_jitters=1)
                        except Exception as e1:
                            print(f"Direct encoding failed: {e1}")
                            encoding = []

                        # If that doesn't work, try with face locations first
                        if not encoding:
                            try:
                                face_locations = face_recognition.face_locations(
                                    face_img)
                                if face_locations:
                                    encoding = face_recognition.face_encodings(
                                        face_img, face_locations, num_jitters=1)
                            except Exception as e2:
                                print(
                                    f"Encoding with face_locations failed: {e2}")
                                encoding = []
                    else:
                        encoding = []

                    # If both approaches fail with the crop, try with the whole image
                    if not encoding:
                        try:
                            # Convert the YOLO box coordinates to face_recognition format (top, right, bottom, left)
                            face_location = [(y1, x2, y2, x1)]
                            encoding = face_recognition.face_encodings(
                                rgb_image, face_location, num_jitters=1)
                        except Exception as e3:
                            print(f"Encoding with original image failed: {e3}")
                            encoding = []

                    if encoding:
                        faces_data.append({
                            'encoding': encoding[0],
                            'location': {
                                'x': x1,
                                'y': y1,
                                'width': x2 - x1,
                                'height': y2 - y1,
                                'confidence': float(conf)
                            }
                        })
                except Exception as e:
                    print(f"Error processing face {i}: {str(e)}")
                    continue

            if not faces_data:
                print("ERROR: Could not extract face encodings")
                return Response({'error': 'Could not extract face features'},
                                status=status.HTTP_400_BAD_REQUEST)

            print(f"Successfully extracted {len(faces_data)} face encodings")

            # Get all stored students
            students = Student.objects.all()
            if not students:
                print("WARNING: No students in database to compare against")
                return Response({'results': [{'student_id': None, 'name': 'Unknown', 'distance': None}]},
                                status=status.HTTP_200_OK)

            known_encodings = [np.frombuffer(s.embedding) for s in students]

            results = []
            newly_marked = []
            already_marked = []
            unknown_faces = 0

            # Match faces
            for face_data in faces_data:
                encoding = face_data['encoding']
                face_location = face_data['location']

                distances = face_recognition.face_distance(
                    known_encodings, encoding)
                if len(distances) == 0:
                    continue

                best_match_index = int(np.argmin(distances))
                best_distance = distances[best_match_index]
                print(
                    f"Best match distance: {best_distance:.4f} (threshold: 0.6)")

                if best_distance < 0.6:  # Lower is better match
                    student = students[best_match_index]
                    # Check if already marked for this session
                    already_exists = AttendanceRecord.objects.filter(
                        student=student,
                        session_id=session_id
                    ).exists()

                    # Create result dictionary with face location
                    result_data = {
                        'student_id': student.student_id,
                        'name': student.name,
                        'distance': float(best_distance),
                        'face_location': face_location,
                        'confidence': face_location['confidence']
                    }

                    if already_exists:
                        print(
                            f"Attendance record already exists for this student and session")
                        result_data['status'] = 'already_marked'
                        already_marked.append(result_data)
                    else:
                        # Create attendance record
                        AttendanceRecord.objects.create(
                            student=student,
                            recognized_by=recognized_by,
                            session_id=session_id,
                        )
                        result_data['status'] = 'newly_marked'
                        newly_marked.append(result_data)
                else:
                    unknown_faces += 1
                    # You can optionally track unknown faces locations

            results.extend(newly_marked)
            results.extend(already_marked)

            return Response({
                'results': results,
                'summary': {
                    'total_faces_detected': len(faces_data),
                    'newly_marked': len(newly_marked),
                    'already_marked': len(already_marked),
                    'unknown_faces': unknown_faces,
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"ERROR during face recognition: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
