from django.conf import settings
from django.shortcuts import render
import os
import cv2
import time
import json
import face_recognition_models
import face_recognition
from django.http import JsonResponse
from datetime import datetime

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
        # Extract existing recognized students
        already_recognized = request.data.get('already_recognized', '')
        recognized_ids = set(already_recognized.split(
            ',')) if already_recognized else set()
        session_id = request.data.get('session_id', 'unknown_session')

        image_file = request.FILES.get('image')
        recognized_by = request.data.get('recognized_by', 'mobile_app')

        # Create a debug folder if it doesn't exist
        debug_folder = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'debug_faces')
        os.makedirs(debug_folder, exist_ok=True)

        if not image_file:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the original uploaded image first
        timestamp = int(time.time())
        filename = f"face_{timestamp}_{recognized_by}.jpg"
        filepath = os.path.join(debug_folder, filename)

        with open(filepath, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        print(f"📸 Saved debug image to: {filepath}")

        # Rewind the file for processing
        image_file.seek(0)

        # Normal processing code...

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
                number_of_times_to_upsample=1
            )

            if len(face_locations) > 10:
                print(
                    f"WARNING: Limiting from {len(face_locations)} faces to 10")
                face_locations = face_locations[:3]

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
                print("No face encodings with standard approach, trying rotations...")
                for angle in [90, 180, 270]:
                    try:
                        print(f"Trying rotation {angle} degrees...")
                        if angle == 90:
                            # Use copy to prevent reference issues
                            rotated = cv2.rotate(
                                image.copy(), cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 180:
                            rotated = cv2.rotate(image.copy(), cv2.ROTATE_180)
                        else:
                            rotated = cv2.rotate(
                                image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)

                        # Significantly reduce detection sensitivity for rotated images
                        rot_face_locations = face_recognition.face_locations(
                            rotated, model="hog", number_of_times_to_upsample=1  # Use 1 instead of 3
                        )

                        # Add strict limit on face count to prevent memory issues
                        if len(rot_face_locations) > 3:  # Much lower than before
                            print(
                                f"Found {len(rot_face_locations)} faces in rotated image - limiting to 3")
                            rot_face_locations = rot_face_locations[:3]
                        elif rot_face_locations:
                            print(
                                f"Found {len(rot_face_locations)} faces in rotated image!")

                        # Add explicit memory cleanup
                        del rotated  # Explicitly release memory

                        # Process only if we have a reasonable number of faces
                        if rot_face_locations and len(rot_face_locations) <= 3:
                            rot_encodings = face_recognition.face_encodings(
                                image,  # Use original image
                                rot_face_locations,
                                num_jitters=1  # Reduce from 3 to 1
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

            # Get all stored students
            students = Student.objects.all()
            if not students:
                print("WARNING: No students in database to compare against")
                return Response({'results': [{'student_id': None, 'name': 'Unknown', 'distance': None}]},
                                status=status.HTTP_200_OK)

            known_encodings = [np.frombuffer(s.embedding) for s in students]

            results = []
            for encoding in encodings:
                distances = face_recognition.face_distance(
                    known_encodings, encoding)
                if len(distances) == 0:
                    continue

                best_match_index = int(np.argmin(distances))
                best_distance = distances[best_match_index]
                print(
                    f"Best match distance: {best_distance:.4f} (threshold: 0.6)")

                # When creating attendance record, check if already done for this session
                if best_distance < 0.67:
                    student = students[best_match_index]

                    # Skip if already recognized in this session
                    if student.student_id in recognized_ids:
                        print(
                            f"Student {student.name} already recognized in this session, skipping attendance")
                    else:
                        # Create attendance record with session ID
                        AttendanceRecord.objects.create(
                            student=student,
                            recognized_by=recognized_by,
                            session_id=session_id
                        )

                    # Always return the student info even if attendance wasn't logged
                    results.append({
                        'student_id': student.student_id,
                        'name': student.name,
                        'distance': float(best_distance),
                        'attendance_logged': student.student_id not in recognized_ids,
                    })
                else:
                    results.append({
                        'student_id': None,
                        'name': 'Unknown',
                        'distance': None
                    })

            # After recognition, save the result alongside the image
            result_filename = f"result_{timestamp}.json"
            result_filepath = os.path.join(debug_folder, result_filename)

            # Save recognition results
            recognition_result = {
                'timestamp': datetime.now().isoformat(),
                'recognized_by': recognized_by,
                'results': results,
                'image_file': filename,
            }

            with open(result_filepath, 'w') as f:
                import json as _json  # Local import to ensure it's available
                _json.dump(recognition_result, f, indent=2)

            # Enhanced debug: save image with face rectangles drawn
            if face_locations:
                debug_image = image.copy()
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    cv2.rectangle(debug_image, (left, top),
                                  (right, bottom), (0, 255, 0), 2)

                rect_filename = f"face_rect_{timestamp}.jpg"
                rect_filepath = os.path.join(debug_folder, rect_filename)
                cv2.imwrite(rect_filepath, debug_image)

            return Response({'results': results}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"ERROR during face recognition: {str(e)}")

            # Save error info
            error_filename = f"error_{timestamp}.txt"
            error_filepath = os.path.join(debug_folder, error_filename)

            with open(error_filepath, 'w') as f:
                f.write(f"Error processing {filename}:\n{str(e)}")

            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def debug_face_viewer(request):
    """Simple viewer for debug face images"""
    debug_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'debug_faces')
    images = []

    # Get all jpg files in the directory
    for filename in sorted(os.listdir(debug_dir), reverse=True):
        if filename.endswith('.jpg'):
            images.append({
                'url': f'/debug-faces/{filename}',
                'name': filename,
                'date': os.path.getmtime(os.path.join(debug_dir, filename))
            })

    return render(request, 'debug_faces.html', {
        'images': images[:100]  # Limit to 100 most recent
    })
