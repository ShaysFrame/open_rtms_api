import os
from django.urls import path
from django.views.static import serve
from . import views

# Assuming your Django project is called 'myproject'
debug_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'debug_faces')
os.makedirs(debug_dir, exist_ok=True)

urlpatterns = [
    path('api/register/', views.RegisterFaceView.as_view(), name='register_face'),
    path('api/recognize/', views.RecognizeFaceView.as_view(), name='recognize_face'),


    # Debug face images (DEBUG only - not for production)
    path('debug-faces/<path:path>',
         serve,
         {'document_root': debug_dir},
         name='debug_faces'),

    # HTML viewer for debug faces
    path('debug-viewer/', views.debug_face_viewer, name='debug_face_viewer'),

]
