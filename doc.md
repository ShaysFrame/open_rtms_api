# Verification & Validation (V&V) Documentation  
## Open RTMS Backend API

## 1. Verification & Validation Overview

The Open RTMS backend API is responsible for securely handling student registration, face recognition, and attendance management. The V&V process ensures that the backend meets functional requirements, handles errors gracefully, and maintains data integrity and security. The primary goals are to verify correct API behavior, validate input/output, and ensure robust error handling under all expected and boundary conditions.

---

## 2. Test Case Specifications

| Test Case ID | Description                              | Preconditions                  | Steps                                                                 | Expected Result                         |
|--------------|------------------------------------------|--------------------------------|-----------------------------------------------------------------------|-----------------------------------------|
| TC-BE-001    | Register student with valid data         | User not registered            | POST `/api/register/` with valid student data and image               | 201 Created, student registered         |
| TC-BE-002    | Register student with missing fields     | -                              | POST `/api/register/` with missing required fields                    | 400 Bad Request, error message          |
| TC-BE-003    | Recognize face with valid image          | Student registered             | POST `/api/recognize/` with valid face image                          | 200 OK, student identified              |
| TC-BE-004    | Recognize face with invalid image        | -                              | POST `/api/recognize/` with corrupted/empty image                     | 400 Bad Request, error message          |
| TC-BE-005    | Mark attendance for recognized student   | Student recognized             | POST `/api/attendance/` with valid student ID                         | 200 OK, attendance marked               |
| TC-BE-006    | API returns error for invalid endpoint   | -                              | GET `/api/unknown/`                                                   | 404 Not Found                           |
| TC-BE-007    | Handle DB connection failure gracefully  | DB unavailable                 | Trigger DB outage, POST `/api/register/`                              | 500 Internal Server Error, error message|
| TC-BE-008    | Rate limit exceeded                     | Multiple rapid requests        | Exceed API rate limit with repeated POSTs                             | 429 Too Many Requests                   |

---

## 3. Boundary Value Test Cases

| Test Case ID | Input                                      | Expected Result                        |
|--------------|--------------------------------------------|----------------------------------------|
| BV-BE-001    | Min field length (e.g., name = 1 char)     | 201 Created or 400 Bad Request         |
| BV-BE-002    | Max field length (e.g., name = 255 chars)  | 201 Created or 400 Bad Request         |
| BV-BE-003    | Image size = 1KB (min)                     | 400 Bad Request (invalid image)        |
| BV-BE-004    | Image size = 5MB (max allowed)             | 201 Created or 400 Bad Request         |
| BV-BE-005    | Exceed max image size (e.g., 10MB)         | 400 Bad Request, error message         |
| BV-BE-006    | API rate limit = 60 requests/min           | 200 OK for first 60, 429 after         |

---

## 4. Formal Verification Methods

- **Code Review:** All API endpoints and business logic are peer-reviewed for correctness, security, and maintainability.
- **Unit Testing:** Automated tests cover serializers, views, and utility functions.
- **Integration Testing:** End-to-end tests validate API workflows and DB interactions.
- **Schema Validation:** Input data is validated using Django REST Framework serializers.
- **Static Analysis:** Tools (e.g., flake8, mypy) are used to detect code issues.
- **Continuous Integration:** Tests are run on every commit via CI pipelines.

---

## 5. Failure Mode Analysis

- **API Errors:** Invalid input, missing fields, or malformed requests return appropriate HTTP error codes and messages.
- **Database Errors:** Connection failures or integrity errors are caught and return 500 errors with logs for debugging.
- **Model Inference Failures:** Face recognition/model errors are handled gracefully, returning 400/500 errors as appropriate.
- **File Handling Errors:** Invalid or empty image uploads are detected and rejected with clear error messages.
- **Rate Limiting:** Excessive requests are throttled and return 429 errors.

---

## 6. FMEA Table Excerpt

| Failure Mode                | Effect                        | Cause                        | Detection                  | Severity | Occurrence | Detection | Mitigation                                   |
|-----------------------------|-------------------------------|------------------------------|----------------------------|----------|------------|-----------|-----------------------------------------------|
| Invalid image upload        | Registration/recognition fail | Corrupt/empty file           | API response, logs         | 7        | 4          | 8         | Validate file, return 400, log error          |
| DB connection failure       | API unavailable               | DB outage/network issue      | API response, monitoring   | 9        | 3          | 7         | Retry logic, alert admin, return 500          |
| Model inference error       | Recognition fails             | Model bug/invalid input      | API response, logs         | 8        | 2          | 7         | Exception handling, return 500, log error     |
| Rate limit exceeded         | API throttled                 | Excessive requests           | API response, monitoring   | 5        | 5          | 9         | Throttle requests, return 429                 |
| Input validation failure    | Data rejected                 | Malformed/missing fields     | API response, logs         | 6        | 6          | 9         | Serializer validation, return 400             |

---

## 7. References

- [Open RTMS Flutter App V&V Documentation](#)
- [Django REST Framework Testing Docs](https://www.django-rest-framework.org/api-guide/testing/)
- [OpenCV Error Handling](https://docs.opencv.org/)