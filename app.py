from flask import Flask, Response, render_template, request, redirect, url_for
import cv2
import mediapipe as mp
import time
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()
hands = mp_hands.Hands()

pTime = 0
video_paused = False  # Global flag to control play/pause

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def generate_frames(video_path):
    global pTime, video_paused
    cap = cv2.VideoCapture(video_path)

    while True:
        if video_paused:
            time.sleep(0.1)
            continue

        success, img = cap.read()

        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pose estimation
        pose_results = pose.process(imgRGB)

        if pose_results.pose_landmarks:
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

            landmarks = pose_results.pose_landmarks.landmark

            def get_landmark_coordinates(landmark_index):
                return [landmarks[landmark_index.value].x * img.shape[1], landmarks[landmark_index.value].y * img.shape[0]]

            angle_definitions = {
            "Left Elbow": (
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            "Right Elbow": (
            mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            "Left Shoulder": (
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            "Right Shoulder": (
            mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            "Left Knee": (
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            "Right Knee": (
            mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            "Left Hip": (
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            "Right Hip": (
            mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            "Left Ankle": (
                mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
            "Right Ankle": (
                mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
        }

            for joint_name, (landmark1, landmark2, landmark3) in angle_definitions.items():
                joint_coordinates = get_landmark_coordinates(landmark2)

                angle = calculate_angle(
                    get_landmark_coordinates(landmark1),
                    get_landmark_coordinates(landmark2),
                    get_landmark_coordinates(landmark3)
                )

                cv2.putText(img, f"{angle:.2f}", tuple(np.array(joint_coordinates).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('video_feed', video_path=filepath))

@app.route('/video_feed/<path:video_path>')
def video_feed(video_path):
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global video_paused
    video_paused = not video_paused
    return ("", 204)

# if __name__ == "__main__":
#     app.run(debug=True)
