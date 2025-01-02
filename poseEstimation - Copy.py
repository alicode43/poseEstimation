import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()
hands = mp_hands.Hands()

cap = cv2.VideoCapture('raj4.mp4')
pTime = 0


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


while True:
    success, img = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pose estimation
    pose_results = pose.process(imgRGB)

    if pose_results.pose_landmarks:
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)  # Red landmarks
        connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                         thickness=1)  # Green connections, thicker lines

        # Draw the landmarks and connections with the specified styles
        mp_drawing.draw_landmarks(
            img,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
        # mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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

        # Hand detection (left hand)
        left_wrist_coords = get_landmark_coordinates(mp_pose.PoseLandmark.LEFT_WRIST)
        hand_results = hands.process(imgRGB)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_coords = [index_finger_tip.x * img.shape[1], index_finger_tip.y * img.shape[0]]

                wrist_hand_angle = calculate_angle(
                    get_landmark_coordinates(mp_pose.PoseLandmark.LEFT_ELBOW),
                    left_wrist_coords,
                    index_finger_tip_coords
                )

                # cv2.putText(img, f"{wrist_hand_angle:.2f}", tuple(np.array(left_wrist_coords).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
