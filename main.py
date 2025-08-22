import cv2
import mediapipe as mp

# Init webcam with high resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Init MediaPipe Face Mesh with iris tracking enabled
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Allow panning OpenCV windows for better display
cv2.namedWindow("Webcam with Eye Boxes", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    left_eye_crop = None
    right_eye_crop = None

    # Process landmarks if a face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            def get_point(index):
                """Get pixel coordinates from normalized landmark."""
                return int(face_landmarks.landmark[index].x * w), int(face_landmarks.landmark[index].y * h)

            # LEFT EYE: Using standard landmarks from MediaPipe
            left_eye_indices = [33, 133, 159, 145]
            left_eye_points = [get_point(i) for i in left_eye_indices]
            lx_min = max(min(pt[0] for pt in left_eye_points) - 20, 0)
            ly_min = max(min(pt[1] for pt in left_eye_points) - 20, 0)
            lx_max = min(max(pt[0] for pt in left_eye_points) + 20, w)
            ly_max = min(max(pt[1] for pt in left_eye_points) + 20, h)

            left_eye_crop = frame[ly_min:ly_max, lx_min:lx_max]
            cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), (0, 0, 255), 2)  # Red box

            # RIGHT EYE
            right_eye_indices = [362, 263, 386, 374]
            right_eye_points = [get_point(i) for i in right_eye_indices]
            rx_min = max(min(pt[0] for pt in right_eye_points) - 20, 0)
            ry_min = max(min(pt[1] for pt in right_eye_points) - 20, 0)
            rx_max = min(max(pt[0] for pt in right_eye_points) + 20, w)
            ry_max = min(max(pt[1] for pt in right_eye_points) + 20, h)

            right_eye_crop = frame[ry_min:ry_max, rx_min:rx_max]
            cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), (0, 255, 0), 2)  # Green box

    # Display left eye crop if available
    if left_eye_crop is not None and left_eye_crop.size > 0:
        left_eye_resized = cv2.resize(left_eye_crop, (300, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Left Eye", left_eye_resized)

    # Display right eye crop if available
    if right_eye_crop is not None and right_eye_crop.size > 0:
        right_eye_resized = cv2.resize(right_eye_crop, (300, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Right Eye", right_eye_resized)

    # Display the main webcam frame with boxes
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Webcam with Eye Boxes", display_frame)

    # Press 'q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup after loop exit
cap.release()
cv2.destroyAllWindows()
