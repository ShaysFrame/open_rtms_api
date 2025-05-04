from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import CustomUser, Attendance
from face_api.face_utils import FaceUtils


class StartAttendance(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Teacher sends image via API to start attendance
        image = request.FILES.get('image')
        detected_user_ids = FaceUtils.process_image(image)

        for user_id in detected_user_ids:
            Attendance.objects.create(user_id=user_id)

        return Response({"status": "Attendance recorded"})


class GetAttendance(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        records = Attendance.objects.filter(user=user)
        return Response({"attendance": records})
