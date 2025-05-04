from django.db import models
from django.contrib.auth.models import AbstractBaseUser, AbstractUser, BaseUserManager, PermissionsMixin


class CustomUser(AbstractUser):
    """
    Custom user model that extends AbstractUser.
    """
    # Class Variables
    # ----------------
    # User role
    ROLE_CHOICES = [
        ("TEACHER", "Teacher"),
        ("STUDENT", "Student"),
    ]

    # Fields
    # ----------------

    # User basic information
    # default login method is phone_number
    phone_number = models.CharField(max_length=15, unique=True)
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=150)

    # User authentication
    password = models.CharField(max_length=128)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    # User profile
    # ----------------
    cover = models.ImageField(
        upload_to='cover_pictures/', null=True, blank=True)
    avatar = models.ImageField(
        upload_to='profile_pictures/', null=True, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)

    # User preferences
    # ----------------
    language = models.CharField(max_length=10, default='en')
    # User settings
    dark_mode = models.BooleanField(default=False)
    # User notifications
    notifications = models.BooleanField(default=True)

    # User activity
    last_login = models.DateTimeField(auto_now=True)

    # User security
    two_factor_auth = models.BooleanField(default=False)

    # User status
    is_online = models.BooleanField(default=False)

    # User biometric data
    # ----------------
    # Store face encoding data for biometric authentication
    # This is a placeholder for the actual encoding data
    # The encoding data should be stored in a secure format
    # and should be encrypted
    # The encoding data should be a binary field
    # The encoding data should be a list of 128 floats
    face_encoding = models.BinaryField(null=True, blank=True)

    # Created at and updated at fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.phone_number} - {self.name}"


class Attendance(models.Model):
    """
    Model to store attendance records.
    """
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.status}"
