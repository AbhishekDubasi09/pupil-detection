import cv2
import mediapipe as mp

# Initialize webcam (HD resolution)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe Face Mesh with iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Allow resizable display windows
cv2.namedWindow("Webcam with Eye Boxes", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL)

# Target size for cropped eye images
eye_display_size = (300, 150)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontally for natural webcam feel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    left_eye_crop = None
    right_eye_crop = None

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:

            def get_point(index):
                """Helper to convert normalized landmark to pixel coordinates."""
                return int(landmarks.landmark[index].x * w), int(landmarks.landmark[index].y * h)

            # Map actual LEFT eye of user (MediaPipe RIGHT eye landmarks)
            left_eye_indices = [362, 263, 386, 374]
            left_eye_points = [get_point(i) for i in left_eye_indices]
            lx_min = max(min(p[0] for p in left_eye_points) - 20, 0)
            ly_min = max(min(p[1] for p in left_eye_points) - 20, 0)
            lx_max = min(max(p[0] for p in left_eye_points) + 20, w)
            ly_max = min(max(p[1] for p in left_eye_points) + 20, h)
            left_eye_crop = frame[ly_min:ly_max, lx_min:lx_max]
            cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), (0, 0, 255), 2)  # Red box

            # Map actual RIGHT eye of user (MediaPipe LEFT eye landmarks)
            right_eye_indices = [33, 133, 159, 145]
            right_eye_points = [get_point(i) for i in right_eye_indices]
            rx_min = max(min(p[0] for p in right_eye_points) - 20, 0)
            ry_min = max(min(p[1] for p in right_eye_points) - 20, 0)
            rx_max = min(max(p[0] for p in right_eye_points) + 20, w)
            ry_max = min(max(p[1] for p in right_eye_points) + 20, h)
            right_eye_crop = frame[ry_min:ry_max, rx_min:rx_max]
            cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), (0, 255, 0), 2)  # Green box

    # Display the cropped left eye (resized for clarity)
    if left_eye_crop is not None and left_eye_crop.size > 0:
        left_eye_resized = cv2.resize(left_eye_crop, eye_display_size, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Left Eye", left_eye_resized)

    # Display the cropped right eye (resized for clarity)
    if right_eye_crop is not None and right_eye_crop.size > 0:
        right_eye_resized = cv2.resize(right_eye_crop, eye_display_size, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Right Eye", right_eye_resized)

    # Show the original frame with eye bounding boxes
    frame_display = cv2.resize(frame, (1280, 720))
    cv2.imshow("Webcam with Eye Boxes", frame_display)

    # Exit condition — press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
