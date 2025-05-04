from django.urls import path
from .views import RegisterFaceView, RecognizeFaceView

urlpatterns = [
    path('api/register/', RegisterFaceView.as_view(), name='register_face'),
    path('api/recognize/', RecognizeFaceView.as_view(), name='recognize_face'),
]
