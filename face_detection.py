import cv2
import dlib
import numpy as np
from scipy.signal import find_peaks

# Initialize face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Helper functions
def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio (EAR) to determine open/closed state."""
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def extract_forehead_region(frame, landmarks):
    """Extract the forehead region for pulse estimation."""
    forehead_region = landmarks[19:27]  # Points above the eyes
    min_x = np.min(forehead_region[:, 0])
    max_x = np.max(forehead_region[:, 0])
    min_y = np.min(forehead_region[:, 1])
    max_y = np.max(forehead_region[:, 1]) - 10
    return frame[min_y:max_y, min_x:max_x]

def estimate_pulse(signal, fps):
    """Estimate pulse from rPPG signal."""
    fft_signal = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
    peaks, _ = find_peaks(fft_signal, height=0.1)
    if peaks.size > 0:
        heart_rate = freqs[peaks[np.argmax(fft_signal[peaks])]] * 60
        return heart_rate
    return None

# Main video capture loop
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

rppg_signal = []  # Stores skin color signal for pulse estimation
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Detect landmarks
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Draw face bounding box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



        # Detect eye states
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        eye_state = "Open" if avg_ear > 0.2 else "Closed"
        cv2.putText(frame, f"Eyes: {eye_state}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Detect gaze direction (basic method using pupils)
        left_pupil_x = np.mean(left_eye[:, 0])
        right_pupil_x = np.mean(right_eye[:, 0])
        gaze_direction = "Center"
        if left_pupil_x < np.mean([left_eye[0][0], left_eye[3][0]]):
            gaze_direction = "Left"
        elif right_pupil_x > np.mean([right_eye[0][0], right_eye[3][0]]):
            gaze_direction = "Right"
        cv2.putText(frame, f"Gaze: {gaze_direction}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Estimate pulse using rPPG
        forehead_region = extract_forehead_region(frame, landmarks)
        avg_color = np.mean(forehead_region, axis=(0, 1))[1]  # Green channel
        rppg_signal.append(avg_color)
        frame_count += 1

        if frame_count >= fps * 10:  # Analyze every 10 seconds
            pulse = estimate_pulse(rppg_signal, fps)
            if pulse:
                cv2.putText(frame, f"Pulse: {int(pulse)} BPM", (x, y + h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rppg_signal.pop(0)  # Reset signal buffer

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
