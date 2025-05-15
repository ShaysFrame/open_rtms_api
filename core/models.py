from django.db import models
from django.utils import timezone


class Student(models.Model):
    name = models.CharField(max_length=255)
    student_id = models.CharField(max_length=50, unique=True)
    embedding = models.BinaryField()  # Stores face embedding as bytes
    photo = models.ImageField(upload_to='students/', null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.student_id})"


class AttendanceRecord(models.Model):
    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name='attendance_records')
    timestamp = models.DateTimeField(default=timezone.now)
    recognized_by = models.CharField(
        max_length=255, blank=True)  # Teacher/device info
    session_id = models.CharField(max_length=100, blank=True)

    class Meta:
        # This ensures only one attendance record per student per session
        unique_together = ['student', 'session_id']

    def __str__(self):
        return f"Attendance: {self.student.name} at {self.timestamp}"
