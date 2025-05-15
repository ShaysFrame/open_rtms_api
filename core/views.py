# Try this alternative import approach
import cv2
import time
import face_recognition_models
import face_recognition

import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Student, AttendanceRecord
from .serializers import StudentSerializer
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile


class RegisterFaceView(APIView):
    def post(self, request):
        name = request.data.get('name')
        student_id = request.data.get('student_id')
        photo = request.FILES.get('photo')

        if not photo or not name or not student_id:
            return Response({'error': 'Missing fields'}, status=status.HTTP_400_BAD_REQUEST)

        # Load and encode the face
        image = face_recognition.load_image_file(photo)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

        embedding = encodings[0].tobytes()

        student = Student.objects.create(
            name=name,
            student_id=student_id,
            photo=photo,
            embedding=embedding
        )
        return Response(StudentSerializer(student).data, status=status.HTTP_201_CREATED)


class RecognizeFaceView(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        recognized_by = request.data.get('recognized_by', 'mobile_app')
        session_id = request.data.get('session_id', 'unknown_session')
        print(f"Session ID: {session_id}")

        print(f"Got recognition request with recognized_by: {recognized_by}")
        print(f"Files in request: {request.FILES.keys()}")

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded, shape: {image.shape}")

            # Save the original image for debugging
            timestamp = int(time.time())
            cv2.imwrite(f"/tmp/original_{timestamp}.jpg",
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Try to enhance the image for better face detection
            enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
            cv2.imwrite(f"/tmp/enhanced_{timestamp}.jpg",
                        cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

            # Make sure image is in the right format for face_recognition
            if image.shape[2] != 3:
                print(
                    f"WARNING: Image has {image.shape[2]} channels, converting to RGB")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # First try with the enhanced image (better to start with enhanced)
            face_locations = face_recognition.face_locations(
                enhanced_image,
                model="hog",
                number_of_times_to_upsample=3  # Even more upsampling
            )

            # If no faces found, try with original image
            if not face_locations:
                print("No faces found in enhanced image, trying original...")
                face_locations = face_recognition.face_locations(
                    image,
                    model="hog",
                    number_of_times_to_upsample=3
                )

            # FIXED: Generate face landmarks and face encodings safely
            encodings = []
            if face_locations:
                print(f"Found {len(face_locations)} face locations!")

                try:
                    # Try getting face encodings directly
                    encodings = face_recognition.face_encodings(
                        image, face_locations)
                except Exception as e:
                    print(
                        f"Error in standard encoding, trying with landmarks: {e}")
                    try:
                        # Alternative approach using face landmarks
                        encodings = []
                        for face_location in face_locations:
                            top, right, bottom, left = face_location
                            # Draw a rectangle on the image to show detected face
                            debug_image = image.copy()
                            cv2.rectangle(debug_image, (left, top),
                                          (right, bottom), (0, 255, 0), 2)
                            cv2.imwrite(f"/tmp/detected_face_{timestamp}.jpg",
                                        cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

                            # Try to get face encoding from the detected face
                            face_encoding = face_recognition.face_encodings(
                                image,
                                [face_location],
                                num_jitters=3  # More jitters = more accuracy
                            )
                            if face_encoding:
                                encodings.append(face_encoding[0])
                    except Exception as e:
                        print(f"Error in alternative encoding approach: {e}")
                        pass

            if not encodings:
                # Try different orientations if no faces found
                print("No face encodings with standard approach, trying rotations...")
                for angle in [90, 180, 270]:
                    try:
                        print(f"Trying rotation {angle} degrees...")
                        if angle == 90:
                            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 180:
                            rotated = cv2.rotate(image, cv2.ROTATE_180)
                        else:
                            rotated = cv2.rotate(
                                image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        cv2.imwrite(f"/tmp/rotated_{angle}_{timestamp}.jpg",
                                    cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))

                        # Get face locations in rotated image
                        rot_face_locations = face_recognition.face_locations(
                            rotated, model="hog", number_of_times_to_upsample=3
                        )

                        if rot_face_locations:
                            print(
                                f"After Rotating to {angle}° Found {len(rot_face_locations)} faces in rotated image!")

                            # Use this method to safely get encodings
                            rot_encodings = face_recognition.face_encodings(
                                rotated,
                                rot_face_locations,
                                num_jitters=3
                            )

                            if rot_encodings:
                                encodings = rot_encodings
                                break
                    except Exception as e:
                        print(f"Error processing rotation {angle}: {e}")
                        continue

            if not encodings:
                print("ERROR: No faces detected in any image orientation")
                return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

            print(f"Successfully generated {len(encodings)} face encodings")

            # Track face locations along with encodings
            faces_data = []
            for i, encoding in enumerate(encodings):
                # We need to preserve which face location corresponds to which encoding
                # If we're using rotated image, the face_locations will be from rot_face_locations
                face_location = face_locations[i] if i < len(
                    face_locations) else None

                if face_location:
                    top, right, bottom, left = face_location
                    faces_data.append({
                        'encoding': encoding,
                        'location': {
                            'x': left,
                            'y': top,
                            'width': right - left,
                            'height': bottom - top,
                        }
                    })
                else:
                    faces_data.append({
                        'encoding': encoding,
                        'location': None
                    })

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

            # Use face_data instead of just encodings
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
                        'face_location': face_location,  # Include face location
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
                    # Optionally, you could track unknown face locations too
                    # unknown_faces_data.append({'face_location': face_location})

            results.extend(newly_marked)
            results.extend(already_marked)

            return Response({'results': results,
                             'summary': {
                                 'total_faces_detected': len(encodings),
                                 'newly_marked': len(newly_marked),
                                 'already_marked': len(already_marked),
                                 'unknown_faces': unknown_faces,
                             }}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"ERROR during face recognition: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
