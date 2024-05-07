from flask import Flask, render_template, Response
import cv2
import numpy as np
import posenet

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def pose_detection():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose detection
        keypoints, _ = posenet.pose(frame)

        # Process keypoints and provide feedback
        feedback = process_keypoints(keypoints)

        # Display feedback on the frame
        cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def process_keypoints(keypoints):
    # Get keypoint positions
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    # Calculate the distance between the left and right shoulders
    shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])

    # Calculate the distance between the left and right wrists
    wrist_distance = abs(left_wrist[0] - right_wrist[0])

    # Set a threshold for acceptable wrist distance
    threshold = 50

    if abs(shoulder_distance - wrist_distance) <= threshold:
        return "Correct bat hold!"
    elif wrist_distance > shoulder_distance:
        return "Hold the bat closer to your body."
    else:
        return "Hold the bat further away from your body."


@app.route('/video_feed')
def video_feed():
    return Response(pose_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
