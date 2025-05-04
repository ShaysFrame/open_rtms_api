from django.http import HttpResponse
from django.shortcuts import render


def test_view(request):
  return HttpResponse("This is a test URL from face_api.")
