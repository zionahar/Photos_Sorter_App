import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuration
FACE_DETECTION_CONFIDENCE = 0.7
FACE_MESH_CONFIDENCE = 0.5
REFINED_LANDMARKS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]  # Iris key points
EYE_CORNER_LANDMARKS = {
    'left_eye': [33, 133],  # Left and right corners of the left eye
    'right_eye': [362, 263]  # Left and right corners of the right eye
}

def calculate_depth(iris_landmarks, frame_width, frame_height):
    """
    Estimate depth based on the size of the iris in pixels.
    Larger iris size indicates closer proximity to the camera.
    """
    left, right, top, bottom = iris_landmarks[:4]
    iris_width = np.linalg.norm(np.array([left[0], left[1]]) - np.array([right[0], right[1]]))
    iris_height = np.linalg.norm(np.array([top[0], top[1]]) - np.array([bottom[0], bottom[1]]))

    # Estimate depth (arbitrary scale for demonstration)
    depth = frame_width / iris_width  # Smaller iris width -> greater depth
    return depth

def estimate_gaze_direction(iris_center, eye_corners):
    """
    Estimate gaze direction as a vector from the center of the iris to the midpoint of eye corners.
    """
    eye_midpoint = np.mean(eye_corners, axis=0)  # Midpoint of the eye corners
    gaze_vector = np.array(iris_center) - eye_midpoint
    gaze_direction = gaze_vector / np.linalg.norm(gaze_vector)  # Normalize the vector
    return gaze_direction

def process_video(video_path=None):
    # Capture video input
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_DETECTION_CONFIDENCE) as face_detection, \
         mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=FACE_MESH_CONFIDENCE, min_tracking_confidence=FACE_MESH_CONFIDENCE) as face_mesh:

        frame_counter = 0
        frame_start_time = time.time()
        while cap.isOpened():
            frame_counter+=1
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # 1. Face Detection
            face_detection_results = face_detection.process(rgb_frame)

            # Draw face detection results
            if face_detection_results.detections:
                for detection in face_detection_results.detections:
                    mp_drawing.draw_detection(frame, detection)

            # 2. Facial Landmark Detection
            face_mesh_results = face_mesh.process(rgb_frame)
            if face_mesh_results.multi_face_landmarks:
                ih, iw, _ = frame.shape  # Get image dimensions
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    # Draw the full face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

                    # Extract iris landmarks
                    iris_points = []
                    for landmark_idx in REFINED_LANDMARKS:
                        landmark = face_landmarks.landmark[landmark_idx]
                        x, y = int(landmark.x * iw), int(landmark.y * ih)
                        iris_points.append((x, y))
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

                    # Extract eye corners
                    eye_corners = {}
                    for eye, corner_indices in EYE_CORNER_LANDMARKS.items():
                        corners = [(int(face_landmarks.landmark[idx].x * iw), int(face_landmarks.landmark[idx].y * ih)) for idx in corner_indices]
                        eye_corners[eye] = corners

                    # Gaze Direction Estimation (for left and right eyes)
                    for eye, corners in eye_corners.items():
                        iris_center = iris_points[0]  # Use the iris center (landmark 468)
                        gaze_direction = estimate_gaze_direction(iris_center, corners)
                        gaze_text = f"Gaze {eye}: {gaze_direction.round(2)}"
                        cv2.putText(frame, gaze_text, (10, 30 if eye == 'left_eye' else 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Depth Estimation
                    depth = calculate_depth(iris_points, iw, ih)
                    cv2.putText(frame, f"Estimated Depth: {depth:.2f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            frame_time = (time.time() - frame_start_time)/frame_counter
            cv2.putText(frame, f"FPS: {1/frame_time}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # Display the output frame
            cv2.imshow('MediaPipe Face and Iris Tracking', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the extended pipeline
process_video()
