# core/serializers.py
from rest_framework import serializers
from .models import Student, AttendanceRecord


class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = ['id', 'name', 'student_id', 'photo']


class AttendanceRecordSerializer(serializers.ModelSerializer):
    student = StudentSerializer(read_only=True)

    class Meta:
        model = AttendanceRecord
        fields = ['id', 'student', 'timestamp', 'recognized_by']
