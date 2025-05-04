from django.urls import path
from django.http import HttpResponse
from .views import test_view

urlpatterns = [
    path('test/', test_view, name='test'),
]
