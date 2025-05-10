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
from rest_framework.decorators import api_view


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


@api_view(['POST'])
def processClassroom(request):
    # Extract data from request
    already_recognized = request.data.get('already_recognized', '')
    recognized_ids = set(already_recognized.split(
        ',')) if already_recognized else set()
    session_id = request.data.get('session_id', 'unknown_session')
    recognized_by = request.data.get('recognized_by', 'Mobile App')

    # Get the image file
    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

    # Create debug folder
    debug_folder = os.path.join(settings.BASE_DIR, 'debug_faces')
    os.makedirs(debug_folder, exist_ok=True)

    # Save the original file for debugging
    timestamp = int(time.time())
    raw_filepath = os.path.join(debug_folder, f"raw_{timestamp}.jpg")

    with open(raw_filepath, 'wb') as f:
        image_file.seek(0)  # Reset file pointer
        for chunk in image_file.chunks():
            f.write(chunk)

    print(
        f"📸 Saved raw image: {raw_filepath}, size: {os.path.getsize(raw_filepath)} bytes")

    # IMPORTANT: Reset file pointer after saving
    image_file.seek(0)

    # Read the file data
    img_bytes = image_file.read()
    print(f"📸 Read {len(img_bytes)} bytes from image file")

    if len(img_bytes) == 0:
        return Response({'error': 'Empty image data'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Convert to numpy array and decode
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return Response({'error': 'Failed to decode image'}, status=status.HTTP_400_BAD_REQUEST)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"📸 Image shape: {image.shape}")

        # Rest of your face detection code...
        face_locations = face_recognition.face_locations(
            rgb_image, model="hog", number_of_times_to_upsample=1)
        print(f"📸 Found {len(face_locations)} faces")

        # Process faces and prepare response...
        face_boxes = []
        recognized_students = []

        for i, face_loc in enumerate(face_locations):
            top, right, bottom, left = face_loc
            face_boxes.append({
                'box': [left, top, right-left, bottom-top],
                'face_index': i
            })

            # Add dummy recognition for now
            recognized_students.append({
                'student_id': None,
                'name': 'Person ' + str(i+1),
                'confidence': 0.8,
                'face_index': i
            })

        # Return data in the format expected by the app
        return Response({
            'face_boxes': face_boxes,
            'recognized_students': recognized_students,
            'total_faces': len(face_locations)
        })

    except Exception as e:
        print(f"Error processing classroom image: {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
